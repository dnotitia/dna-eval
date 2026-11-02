"""``weighted_accuracy``: accuracy in which each sample counts as much as its weight."""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric


@metric
def weighted_accuracy(weight_key: str = "score") -> Metric:
    """Mean score weighted by the sample metadata field ``weight_key``: ``sum(w * s) / sum(w)``.

    A sample without the field counts with weight 1; NaN when the weights sum to zero.
    """

    def compute(scores: list[SampleScore]) -> Value:
        pairs = [(s.score.as_float(), float((s.sample_metadata or {}).get(weight_key, 1.0))) for s in scores]
        total = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total if total else float("nan")

    return compute
