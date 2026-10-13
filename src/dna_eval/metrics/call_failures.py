"""``call_failures``: how often a score stands for a failed call rather than a judged answer.

A failed call does not drop the sample: no answer from the model becomes a placeholder that
is judged like an answer, and a judgment that still fails after its retries scores 0. Both
keep every average over the same samples, and both pull the score down; this metric says by
how much, so a low number can be told apart from a flaky endpoint.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata keys read here.
NO_ANSWER = "no_answer"        # "failed" (every call raised) or "empty" (no answer text); absent otherwise
JUDGE_ERROR = "judge_error"    # "call" or "parse" after the last retry; absent otherwise
COMPONENTS = "components"      # composite rubrics: per-component metadata with "judge_error"


def _judge_failed(md: dict) -> bool:
    if md.get(JUDGE_ERROR):
        return True
    return any((c or {}).get(JUDGE_ERROR) for c in (md.get(COMPONENTS) or {}).values())


@metric(scores="unreduced")
def call_failures() -> Metric:
    """Shares of all scored answers, every run counted (``scores="unreduced"``):

    - ``answer_failed``: every call to the model under evaluation raised (timeouts, HTTP errors);
    - ``answer_empty``: the model returned, but with no answer text (e.g. only reasoning);
    - ``judge_failed``: the judge call or its verdict failed after every retry (scored 0).
    """

    def compute(scores: list[SampleScore]) -> Value:
        n = len(scores)
        if n == 0:
            return {"answer_failed": 0.0, "answer_empty": 0.0, "judge_failed": 0.0}
        md = [s.score.metadata or {} for s in scores]
        return {
            "answer_failed": sum(m.get(NO_ANSWER) == "failed" for m in md) / n,
            "answer_empty": sum(m.get(NO_ANSWER) == "empty" for m in md) / n,
            "judge_failed": sum(_judge_failed(m) for m in md) / n,
        }

    return compute
