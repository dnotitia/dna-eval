"""Per-run statistics must match the llm-model-test convention."""
from inspect_ai.scorer import SampleScore, Score

from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.utils.stats import trim_mean


def test_trim_mean_integer_truncation():
    # 15 values: int(15 * 0.15) = 2 dropped from each end -> mean of 3..13
    assert trim_mean([float(i) for i in range(1, 16)]) == 8.0
    assert trim_mean([1.0, 2.0, 3.0]) == 2.0          # k = 0: nothing dropped


def test_per_run_trimmed_groups_by_epoch_and_trusts_passed():
    scores = []
    for epoch in range(1, 4):
        for i in range(4):
            v = 1.0 if (i + epoch) % 2 else 0.5         # two pass, two fail per run
            scores.append(SampleScore(score=Score(value=v, metadata={"epoch": epoch, "passed": v >= 0.7}), sample_id=i))
    out = per_run_trimmed()(scores)
    assert out["runs"] == 3
    assert abs(out["trimmed_mean"] - 0.75) < 1e-9
    assert out["trimmed_passed"] == 2.0
