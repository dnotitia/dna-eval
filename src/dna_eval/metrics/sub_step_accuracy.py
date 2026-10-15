"""``sub_step_accuracy``: the share of sub-steps that passed their tests, over all runs."""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata keys a benchmark scorer sets for this metric.
STEPS_PASSED = "steps_passed"   # int: sub-steps of the problem whose tests all passed
STEPS_TOTAL = "steps_total"     # int: sub-steps of the problem that are evaluated


@metric(scores="unreduced")
def sub_step_accuracy() -> Metric:
    """Sub-steps passed divided by sub-steps evaluated, summed over every scored sample.

    The metric sees every epoch of every sample (``scores="unreduced"``), so with ``epochs > 1``
    the sub-steps of all runs are pooled: each run of a problem counts its own sub-steps, and
    problems with more sub-steps weigh more than those with fewer. NaN with no scores.
    """

    def compute(scores: list[SampleScore]) -> Value:
        md = [s.score.metadata or {} for s in scores]
        total = sum(int(m.get(STEPS_TOTAL, 0)) for m in md)
        return sum(int(m.get(STEPS_PASSED, 0)) for m in md) / total if total else float("nan")

    return compute
