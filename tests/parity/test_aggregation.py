"""Per-run trimmed mean and pass count reproduce the values recorded from a DeepEval GEval evaluation."""
import json
from pathlib import Path

import pytest
from inspect_ai.scorer import SampleScore, Score

from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.utils.stats import trim_mean
from tests.parity.cases import CASES

FIXTURES = Path(__file__).parent.parent / "fixtures" / "deepeval_runs"


@pytest.mark.parametrize("task", list(CASES))
def test_trim_mean_reproduces_deepeval(task):
    fx = json.loads((FIXTURES / f"{task}.json").read_text(encoding="utf-8"))
    assert round(trim_mean([r["mean"] for r in fx["runs"]]), 4) == fx["expected"]["trimmed_mean"]
    assert round(trim_mean([r["passed"] for r in fx["runs"]]), 2) == fx["expected"]["trimmed_passed"]


@pytest.mark.parametrize("task", list(CASES))
def test_per_run_trimmed_reproduces_deepeval(task):
    """Rebuild per-sample scores whose per-run means/pass counts equal the DeepEval runs, then
    check the in-pipeline metric lands on the same trimmed statistics."""
    fx = json.loads((FIXTURES / f"{task}.json").read_text(encoding="utf-8"))
    n = 100  # synthetic samples per run; only the per-run mean and pass count matter
    scores = []
    for epoch, run in enumerate(fx["runs"], start=1):
        # `passed` samples score 1.0, the rest share the remainder so the run mean is exact
        k = round(run["passed"] / max(r["passed"] for r in fx["runs"]) * n) if False else min(n, int(run["passed"]))
        rest = (run["mean"] * n - k) / (n - k) if n > k else 0.0
        vals = [1.0] * k + [rest] * (n - k)
        for i, v in enumerate(vals):
            scores.append(SampleScore(score=Score(value=v, metadata={"epoch": epoch, "passed": i < k}), sample_id=i))
    out = per_run_trimmed()(scores)
    assert out["runs"] == len(fx["runs"])
    assert round(out["trimmed_mean"], 4) == fx["expected"]["trimmed_mean"]
    assert round(out["trimmed_passed"], 2) == fx["expected"]["trimmed_passed"]
