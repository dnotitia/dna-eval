"""``rubric_scorer``: judge an answer against a rubric, reported under the rubric's name.

Every rubric shipped in ``rubrics/`` is registered with Inspect as a scorer of the same name
when this module is imported (``dna_eval/performance``, ``dna_eval/uncensorship``, ...). The
name is what the log viewer shows, and what ``inspect score`` looks up to re-score a log, for
example with a different judge::

    inspect score run.eval --scorer dna_eval/repetition --action overwrite --model-role grader=<judge>

A judge rubric is one judge pass. A composite rubric (``compose``) judges the same answer
against each of its components and folds them into one score; see ``rubrics/__init__.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from inspect_ai.model import ContentReasoning, GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from pydantic import BaseModel

from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.prompts import build_prompt
from dna_eval.rubrics import CompositeRubric, Rubric, load_rubric, rubric_names
from dna_eval.solvers.dna_generate import REASONING_KEY, reasoning_text
from dna_eval.utils.json import extract_json


class _Verdict(BaseModel):
    """DeepEval's ``ReasonScore``: both fields required, ``score`` coerced to float."""
    reason: str
    score: float


def read_verdict(reply: str, rubric: Rubric) -> tuple[float, float, str]:
    """(normalised score, raw score, reason) from a judge reply, read as DeepEval's GEval reads it.

    Raises ``ValueError`` (``pydantic.ValidationError`` included) when the reply cannot be
    read. The score is not clipped to the rubric range, as in DeepEval.
    """
    v = _Verdict.model_validate(extract_json(reply))
    return (v.score - rubric.lo) / (rubric.hi - rubric.lo), v.score, v.reason


def content_as_sent(output: ModelOutput) -> str:
    """The judge message's ``content`` as the server sent it, which is what DeepEval reads.

    Inspect moves a ``<think>...</think>`` block out of the text into a ``ContentReasoning``
    (``internal="think"``). DeepEval parses the raw content, so a brace inside that block
    makes it fail; put the block back so ``read_verdict`` fails the same way. Reasoning sent
    in a separate field (``reasoning_content``) is left out, as DeepEval never sees it.
    """
    text = output.completion
    if not output.choices or isinstance(output.message.content, str) or text.lstrip().startswith("<think"):
        return text
    think = "".join(f"<think>{c.reasoning}</think>" for c in output.message.content
                    if isinstance(c, ContentReasoning) and c.internal == "think")
    return think + text


@dataclass
class _Judgment:
    rubric: Rubric
    value: float = 0.0
    raw: float | None = None
    reason: str = ""
    error: str = ""          # "", "parse" or "call"

    @property
    def passed(self) -> bool:
        return not self.error and self.value >= self.rubric.threshold


def _field(r: Rubric, param: str, state: TaskState, target: Target, context: list[str] | None):
    """The value of judge-visible field ``param``, from where the rubric's ``sources`` say."""
    src = r.source(param)
    if src == "input":
        return state.input_text
    if src == "answer":
        return state.output.completion
    if src == "reasoning":                       # set by dna_generate; derived if another solver ran
        return state.store.get(REASONING_KEY) or reasoning_text(state.output)
    if src == "target":
        return target.text
    value = state.metadata.get(src.removeprefix("metadata."))
    return context if value is None and param == "context" else value


async def _judge_once(r: Rubric, state: TaskState, target: Target, context: list[str] | None) -> _Judgment:
    """One judge call for one rubric; a failed call is returned, not raised."""
    f = lambda p: _field(r, p, state, target, context)  # noqa: E731
    prompt = build_prompt(r, input_text=f("input"), actual_output=f("actual_output"),
                          expected_output=f("expected_output") or "", context=f("context"))
    grader = get_model(role="grader", required=True, config=GenerateConfig(temperature=0.0))
    last = ""
    try:
        last = content_as_sent(await grader.generate(prompt))
        value, raw, reason = read_verdict(last, r)
        return _Judgment(r, value, raw, reason)
    except ValueError:
        return _Judgment(r, reason=f"judge failed: {last[:300]}", error="parse")
    except Exception as e:  # noqa: BLE001  (any failed call is retried)
        return _Judgment(r, reason=f"judge failed: {type(e).__name__}: {e}"[:300], error="call")


async def _judge_rounds(parts: list[Rubric], state: TaskState, target: Target,
                        judge_retries: int, context: list[str] | None) -> list[_Judgment]:
    """Judge every rubric; if any judgment failed, judge them all again, up to ``judge_retries``
    more rounds (as DeepEval re-evaluates a whole failed test case). Judgments still failing
    after the last round score 0, so the sample still counts in every average."""
    for _ in range(1 + judge_retries):
        judgments = [await _judge_once(r, state, target, context) for r in parts]
        if not any(j.error for j in judgments):
            break
    return judgments


def _judge_scorer(rubric: Rubric | CompositeRubric, judge_retries: int, context: list[str] | None) -> Scorer:
    parts = rubric.components() if isinstance(rubric, CompositeRubric) else [rubric]

    async def score(state: TaskState, target: Target) -> Score:
        js = await _judge_rounds(parts, state, target, judge_retries, context)
        if not isinstance(rubric, CompositeRubric):
            j = js[0]
            md = {"passed": j.passed, "epoch": state.epoch}
            md |= {"judge_error": j.error} if j.error else {"raw_score": j.raw}
            return Score(value=j.value, answer=state.output.completion, explanation=j.reason, metadata=md)
        return Score(
            value=fmean(j.value for j in js),                       # combine: mean
            answer=state.output.completion,
            explanation="\n\n".join(f"[{j.rubric.name}] {j.reason}" for j in js),
            metadata={
                "passed": all(j.passed for j in js),               # pass: all
                "epoch": state.epoch,
                "components": {j.rubric.name: {"value": j.value, "passed": j.passed, "raw": j.raw,
                                               "judge_error": j.error or None} for j in js},
            },
        )

    return score


def _register(source: str, name: str):
    """Register the scorer for one rubric with Inspect, under ``name``."""

    def factory(judge_retries: int = 2, context: list[str] | None = None) -> Scorer:
        """Judge ``state.output`` against the rubric with the model bound to the ``grader`` role.

        The judge is called at temperature 0 and is expected to answer ``{"score": lo-hi,
        "reason": ...}``. The score is normalised to 0-1; ``metadata["passed"]`` applies the
        rubric threshold (for a composite: every component passes); ``metadata["epoch"]``
        lets ``per_run_trimmed`` regroup scores by run. A failed or unreadable judge call is
        retried ``judge_retries`` times (for a composite: all components are judged again)
        and then scored 0, so the sample still counts in every average.

        ``context`` is shown to the judge for every sample, for a rubric whose ``params``
        include ``context``; a sample's own ``metadata["context"]`` takes precedence. Passing
        it here keeps a long shared document (e.g. a persona sheet) out of every sample.
        """
        return _judge_scorer(load_rubric(source), judge_retries, context)

    return scorer(metrics=[per_run_trimmed(), mean()], name=name)(factory)


_SCORERS = {name: _register(name, name) for name in rubric_names()}


def rubric_scorer(rubric: str, judge_retries: int = 2, context: list[str] | None = None) -> Scorer:
    """The scorer for ``rubric``: a rubric name from ``rubrics/``, or a path to a rubric YAML.

    A rubric given by path is registered under its ``name`` field when this is called, so a
    log scored with it can be re-scored only in a process that calls this again (a task
    module that builds it on import, or ``--scorer path/to/module.py``).
    """
    if rubric not in _SCORERS:
        r = load_rubric(rubric)
        if r.name in _SCORERS and rubric != r.name:
            raise ValueError(f"rubric {rubric!r} is named {r.name!r}, which a shipped rubric already uses")
        _SCORERS[rubric] = _register(rubric, r.name)
    return _SCORERS[rubric](judge_retries=judge_retries, context=context)
