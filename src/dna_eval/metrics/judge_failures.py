"""``judge_failures``: how often a score is a judgment that never came back.

A judge call that fails, or whose reply cannot be read, is retried; one still failing after
the last retry scores 0 and stays in every average. This metric gives the share of such
scores, so a low number can be told apart from an unreliable judge endpoint.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata keys read here; every judge-graded scorer sets ``JUDGE_ERROR`` on a failure.
JUDGE_ERROR = "judge_error"    # "call" or "parse" after the last retry; absent otherwise
COMPONENTS = "components"      # composite rubrics: per-component metadata with ``JUDGE_ERROR``


def judge_failed(md: dict) -> bool:
    """Whether a score's metadata records a failed judgment (any component, for a composite)."""
    if md.get(JUDGE_ERROR):
        return True
    return any((c or {}).get(JUDGE_ERROR) for c in (md.get(COMPONENTS) or {}).values())


@metric(scores="unreduced")
def judge_failures() -> Metric:
    """``judge_failed``: the share of scored answers, every run counted (``scores="unreduced"``),
    whose judgment failed after every retry and was scored 0."""

    def compute(scores: list[SampleScore]) -> Value:
        if not scores:
            return {"judge_failed": 0.0}
        return {"judge_failed": sum(judge_failed(s.score.metadata or {}) for s in scores) / len(scores)}

    return compute
