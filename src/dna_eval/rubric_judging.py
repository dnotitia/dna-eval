"""The rubric judge: calling it with a rubric and reading its verdict.

``prompts`` renders what the judge reads; this module makes the call, reads the reply the way
DeepEval's GEval reads it, and runs the re-judge rounds. ``scorers/rubric_scorer.py`` turns the
judgments into Inspect scores.
"""
from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from pydantic import BaseModel

from dna_eval.prompts import build_prompt
from dna_eval.roles import GRADER_ROLE
from dna_eval.rubrics import Rubric
from dna_eval.utils.json import escape_stray_backslashes, extract_json
from dna_eval.utils.model_output import content_as_sent, reasoning_text


class _Verdict(BaseModel):
    """DeepEval's ``ReasonScore``: both fields required, ``score`` coerced to float."""
    reason: str
    score: float


def read_verdict(reply: str, rubric: Rubric) -> tuple[float, float, str]:
    """(normalised score, raw score, reason) from a judge reply, read as DeepEval's GEval reads it,
    with one exception (decision record 0015): a reply DeepEval cannot parse is read again with
    its stray backslashes escaped, so a verdict quoting LaTeX (``\\pi``) is not lost.

    Raises ``ValueError`` (``pydantic.ValidationError`` included) when the reply cannot be
    read. The score is not clipped to the rubric range, as in DeepEval.
    """
    try:
        data = extract_json(reply)
    except ValueError:
        data = extract_json(escape_stray_backslashes(reply))
    v = _Verdict.model_validate(data)
    return (v.score - rubric.lo) / (rubric.hi - rubric.lo), v.score, v.reason


@dataclass
class Judgment:
    """One judge pass: the normalised score, or the reason it failed."""
    rubric: Rubric
    value: float = 0.0
    raw: float | None = None
    reason: str = ""
    error: str = ""          # "", "parse" or "call"

    @property
    def passed(self) -> bool:
        return not self.error and self.value >= self.rubric.threshold


def judge_field(r: Rubric, param: str, state: TaskState, target: Target, context: list[str] | None):
    """The value of judge-visible field ``param``, from where the rubric's ``sources`` say."""
    src = r.source(param)
    if src == "input":
        return state.input_text
    if src == "answer":
        return state.output.completion
    if src == "reasoning":
        return reasoning_text(state.output)
    if src == "target":
        return target.text
    if src.startswith("store."):
        return state.store.get(src.removeprefix("store."))
    value = state.metadata.get(src.removeprefix("metadata."))
    return context if value is None and param == "context" else value


async def judge_once(r: Rubric, state: TaskState, target: Target, context: list[str] | None) -> Judgment:
    """One judge call for one rubric, with the model bound to the ``grader`` role (required).
    A failed or unreadable call is returned as a failed ``Judgment``, not raised.

    The verdict is read from the reply content as the server sent it (``content_as_sent``):
    DeepEval parses the raw content, so a ``<think>`` block Inspect split off must count.
    """
    f = lambda p: judge_field(r, p, state, target, context)  # noqa: E731
    prompt = build_prompt(r, input_text=f("input"), actual_output=f("actual_output"),
                          expected_output=f("expected_output") or "", context=f("context"))
    grader = get_model(role=GRADER_ROLE, required=True, config=GenerateConfig(temperature=0.0))
    last = ""
    try:
        last = content_as_sent(await grader.generate(prompt))
        value, raw, reason = read_verdict(last, r)
        return Judgment(r, value, raw, reason)
    except ValueError:
        return Judgment(r, reason=f"judge failed: {last[:300]}", error="parse")
    except Exception as e:  # noqa: BLE001  (any failed call is retried)
        return Judgment(r, reason=f"judge failed: {type(e).__name__}: {e}"[:300], error="call")


async def judge_rounds(parts: list[Rubric], state: TaskState, target: Target,
                       judge_retries: int, context: list[str] | None) -> list[Judgment]:
    """Judge every rubric; if any judgment failed, judge them all again, up to ``judge_retries``
    more rounds (as DeepEval re-evaluates a whole failed test case). Judgments still failing
    after the last round score 0, so the sample still counts in every average."""
    for _ in range(1 + judge_retries):
        judgments = [await judge_once(r, state, target, context) for r in parts]
        if not any(j.error for j in judgments):
            break
    return judgments
