"""``call_failures``: how often the model under evaluation gave no answer.

A failed call does not drop the sample: ``dna_generate`` retries it, and if no answer comes
back the judge sees a placeholder and scores it like any answer. That keeps every average over
the same samples and pulls the score down; this metric says by how much, so a low number can be
told apart from a flaky endpoint. It reads what ``dna_generate`` records, so it applies to tasks
that generate with it. Failed judgments are counted by ``judge_failures``.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata key read here.
NO_ANSWER = "no_answer"        # "failed" (every call raised) or "empty" (no answer text); absent otherwise


@metric(scores="unreduced")
def call_failures() -> Metric:
    """Shares of all scored answers, every run counted (``scores="unreduced"``):

    - ``answer_failed``: every call to the model under evaluation raised (timeouts, HTTP errors);
    - ``answer_empty``: the model returned, but with no answer text (e.g. only reasoning).
    """

    def compute(scores: list[SampleScore]) -> Value:
        n = len(scores)
        if n == 0:
            return {"answer_failed": 0.0, "answer_empty": 0.0}
        md = [s.score.metadata or {} for s in scores]
        return {
            "answer_failed": sum(m.get(NO_ANSWER) == "failed" for m in md) / n,
            "answer_empty": sum(m.get(NO_ANSWER) == "empty" for m in md) / n,
        }

    return compute
