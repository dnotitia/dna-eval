"""``macro_accuracy``: the mean of the per-group accuracies, each group counting equally."""
from __future__ import annotations

from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, Value, metric


@metric
def macro_accuracy(group_key: str) -> Metric:
    """Accuracy within each value of the sample metadata field ``group_key``, then the plain mean
    of those accuracies; NaN when there is no score."""

    def compute(scores: list[SampleScore]) -> Value:
        groups: dict[str, list[float]] = defaultdict(list)
        for s in scores:
            groups[str((s.sample_metadata or {}).get(group_key))].append(s.score.as_float())
        return sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else float("nan")

    return compute
