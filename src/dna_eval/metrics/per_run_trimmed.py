"""``per_run_trimmed``: the suite's per-run statistics, computed inside the pipeline.

Inspect's reducers fold the epochs of *one sample*. The suite convention folds the other
way: average the dataset within each run (epoch), then take a 15% trimmed mean of those
per-run means, plus the trimmed mean of per-run pass counts. ``@metric(scores="unreduced")``
hands this metric one ``SampleScore`` per sample per epoch; the scorer stamps ``epoch`` and
``passed`` into ``Score.metadata``, so the per-run grouping is rebuilt here and the rubric's
threshold stays the single source of truth for "passed".
"""
from __future__ import annotations

from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, Value, metric

from dna_eval.utils.stats import trim_mean


@metric(scores="unreduced")
def per_run_trimmed(proportiontocut: float = 0.15) -> Metric:
    """``trimmed_mean`` of per-run dataset means, ``trimmed_passed`` of per-run pass counts, ``runs``."""

    def compute(scores: list[SampleScore]) -> Value:
        by_run: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        for s in scores:
            md = s.score.metadata or {}
            if md.get("epoch") is None:
                continue
            by_run[int(md["epoch"])].append((s.score.as_float(), bool(md.get("passed", False))))
        if not by_run:
            return {"trimmed_mean": float("nan"), "trimmed_passed": float("nan"), "runs": 0}
        means = [sum(x for x, _ in v) / len(v) for v in by_run.values()]
        passed = [float(sum(1 for _, ok in v if ok)) for v in by_run.values()]
        return {
            "trimmed_mean": trim_mean(means, proportiontocut),
            "trimmed_passed": trim_mean(passed, proportiontocut),
            "runs": len(by_run),
        }

    return compute
