"""``pass_hat_k``: tau-bench's pass^k, the chance that an agent solves a task in all of ``k`` trials.

Defined in the tau-bench paper (arXiv:2406.12045) and computed by tau2-bench's
``metrics/agent_metrics.py:pass_hat_k``: for a task solved in ``c`` of ``n`` trials,
``C(c, k) / C(n, k)``, averaged over tasks. ``pass^1`` is the mean success rate. A trial is a
success when its score is 1 (within 1e-6, as tau2's ``is_successful``).
"""
from __future__ import annotations

import math
from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, Value, metric


@metric(scores="unreduced")
def pass_hat_k() -> Metric:
    """``pass_hat_1`` ... ``pass_hat_<n>``, where ``n`` is the smallest number of trials any task
    has (every run counts: ``scores="unreduced"``, grouped by sample id). A task with more trials
    than ``n`` contributes its first ``n``."""

    def compute(scores: list[SampleScore]) -> Value:
        trials: dict[object, list[bool]] = defaultdict(list)
        for s in scores:
            trials[s.sample_id].append(abs(s.score.as_float() - 1.0) <= 1e-6)
        if not trials:
            return {}
        n = min(len(v) for v in trials.values())
        wins = [sum(v[:n]) for v in trials.values()]
        return {f"pass_hat_{k}": sum(math.comb(c, k) / math.comb(n, k) for c in wins) / len(wins)
                for k in range(1, n + 1)}

    return compute
