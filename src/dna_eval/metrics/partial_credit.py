"""``partial_credit``: the mean share of a task's checks that the agent satisfied.

Some agent benchmarks grade a run by a list of checks on the final state of the environment,
and report both a strict pass (every check holds) and a partial credit (the share that hold).
The scorer writes the second number to ``Score.metadata["partial_credit"]``; this metric
averages it, so the dense number can be read next to the strict pass rate.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# ``Score.metadata`` key a scorer writes for this metric: a float in [0, 1].
PARTIAL_CREDIT = "partial_credit"


@metric(scores="unreduced")
def partial_credit() -> Metric:
    """The mean of ``Score.metadata["partial_credit"]`` over every scored (sample, epoch);
    ``nan`` when no score carries one. Every run counts once (``scores="unreduced"``)."""

    def compute(scores: list[SampleScore]) -> Value:
        values = [m[PARTIAL_CREDIT] for s in scores
                  if isinstance(m := s.score.metadata or {}, dict) and isinstance(m.get(PARTIAL_CREDIT), (int, float))]
        return sum(values) / len(values) if values else float("nan")

    return compute
