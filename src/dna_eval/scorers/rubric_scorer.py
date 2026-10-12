"""``rubric_scorer``: judge an answer against a rubric, reported under the rubric's name.

Every rubric shipped in ``rubrics/`` is registered with Inspect as a scorer of the same name
when this module is imported (``dna_eval/performance``, ``dna_eval/uncensorship``, ...). The
name is what the log viewer shows, and what ``inspect score`` looks up to re-score a log, for
example with a different judge::

    inspect score run.eval --scorer dna_eval/repetition --action overwrite --model-role grader=<judge>

A judge rubric is one judge pass. A composite rubric (``compose``) judges the same answer
against each of its components and folds them into one score; see ``rubrics/__init__.py``.
The judge calls themselves are in ``judging.py``.
"""
from __future__ import annotations

from statistics import fmean

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from dna_eval.judging import judge_rounds
from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.rubrics import CompositeRubric, Rubric, load_rubric, rubric_names


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
        return _score(load_rubric(source), judge_retries, context)

    return scorer(metrics=[per_run_trimmed(), mean()], name=name)(factory)


def _score(rubric: Rubric | CompositeRubric, judge_retries: int, context: list[str] | None) -> Scorer:
    parts = rubric.components() if isinstance(rubric, CompositeRubric) else [rubric]

    async def score(state: TaskState, target: Target) -> Score:
        js = await judge_rounds(parts, state, target, judge_retries, context)
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
