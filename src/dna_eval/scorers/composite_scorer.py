"""``composite``: run several scorers on one generation and fold them into one score."""
from __future__ import annotations

from statistics import fmean

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from dna_eval.metrics.per_run_trimmed import per_run_trimmed


@scorer(metrics=[mean(), per_run_trimmed()])
def composite_scorer(scorers: list[Scorer], names: list[str]) -> Scorer:
    """Score = mean of component scores; passed = every component passed (DeepEval calculate_stats).

    Components are plain Inspect scorers (e.g. several ``rubric_scorer``); each is awaited
    on the same ``state`` so the generation happens once. Component results are kept in
    ``metadata["components"]`` for inspection; ``epoch`` is copied from the first component.
    """
    assert len(scorers) == len(names)

    async def score(state: TaskState, target: Target) -> Score:
        parts = [await s(state, target) for s in scorers]
        md = [p.metadata or {} for p in parts]
        return Score(
            value=fmean(p.as_float() for p in parts),
            answer=" / ".join(p.answer or "-" for p in parts),
            explanation="\n\n".join(f"[{n}] {p.explanation or ''}" for n, p in zip(names, parts)),
            metadata={
                "passed": all(m.get("passed", False) for m in md),
                "epoch": md[0].get("epoch", state.epoch),
                "components": {n: {"value": p.as_float(), "passed": m.get("passed"), "raw": m.get("raw_score")}
                               for n, p, m in zip(names, parts, md)},
            },
        )

    return score