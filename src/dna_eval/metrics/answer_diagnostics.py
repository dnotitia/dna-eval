"""``answer_diagnostics``: how often answers were cut off or missing their answer format."""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata keys a benchmark scorer sets for this metric.
TRUNCATED = "truncated"      # bool: generation hit a token or context limit
EXTRACTED = "extracted"      # the answer read from the response, or None if none was found


@metric(scores="unreduced")
def answer_diagnostics() -> Metric:
    """``truncated`` and ``format_failure`` rates over every scored answer, and
    ``accuracy_excl_truncated``: the mean score of the answers that were not cut off.

    The metric sees every epoch of every sample (``scores="unreduced"``), so with
    ``epochs > 1`` each (sample, epoch) answer counts once; a reduced score would carry
    the metadata of one epoch only.

    Truncated and format-failing answers score 0 in the headline accuracy; these values
    show how much of the gap they explain, and let numbers that exclude truncation be
    compared too.
    """

    def compute(scores: list[SampleScore]) -> Value:
        n = len(scores)
        if n == 0:
            return {"truncated": 0.0, "format_failure": 0.0, "accuracy_excl_truncated": float("nan")}
        md = [s.score.metadata or {} for s in scores]
        truncated = [bool(m.get(TRUNCATED)) for m in md]
        kept = [s.score.as_float() for s, t in zip(scores, truncated) if not t]
        return {
            "truncated": sum(truncated) / n,
            "format_failure": sum(1 for m in md if m.get(EXTRACTED) is None) / n,
            "accuracy_excl_truncated": sum(kept) / len(kept) if kept else float("nan"),
        }

    return compute
