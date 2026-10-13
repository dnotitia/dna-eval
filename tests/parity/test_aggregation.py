"""Per-run trimmed mean and pass count reproduce the values recorded from a DeepEval GEval evaluation.

A fixture holds one block of runs (``runs`` / ``expected``) per scorer: at the top level for a
single-scorer task, under ``scorers.<name>`` for a task with several independent scorers.
"""
import json
from pathlib import Path

import pytest
from inspect_ai.scorer import SampleScore, Score

from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.utils.stats import trim_mean
from tests.parity.cases import CASES

FIXTURES = Path(__file__).parent.parent / "fixtures" / "deepeval_runs"


def _fixture(task: str) -> dict:
    return json.loads((FIXTURES / f"{task}.json").read_text(encoding="utf-8"))


def _blocks() -> dict[str, dict]:
    out = {}
    for task in CASES:
        fx = _fixture(task)
        if "scorers" in fx:
            out |= {f"{task}:{name}": block for name, block in fx["scorers"].items()}
        else:
            out[task] = fx
    return out


BLOCKS = _blocks()


def _synthetic_scores(runs: list[dict], n: int, **sample_kw) -> list[SampleScore]:
    """Per-sample scores whose per-run mean and pass count equal the recorded run:
    ``passed`` samples score 1.0 and the rest share the remainder."""
    scores = []
    for epoch, run in enumerate(runs, start=1):
        k = min(n, int(run["passed"]))
        rest = (run["mean"] * n - k) / (n - k) if n > k else 0.0
        for i, v in enumerate([1.0] * k + [rest] * (n - k)):
            scores.append(SampleScore(score=Score(value=v, metadata={"epoch": epoch, "passed": i < k}),
                                      sample_id=f"{sample_kw.get('prefix', '')}{i}",
                                      sample_metadata=sample_kw.get("metadata")))
    return scores


@pytest.mark.parametrize("block", list(BLOCKS))
def test_trim_mean_reproduces_deepeval(block):
    b = BLOCKS[block]
    assert round(trim_mean([r["mean"] for r in b["runs"]]), 4) == b["expected"]["trimmed_mean"]
    assert round(trim_mean([r["passed"] for r in b["runs"]]), 2) == b["expected"]["trimmed_passed"]


@pytest.mark.parametrize("block", list(BLOCKS))
def test_per_run_trimmed_reproduces_deepeval(block):
    """The in-pipeline metric lands on the same trimmed statistics as the recorded runs."""
    b = BLOCKS[block]
    out = per_run_trimmed()(_synthetic_scores(b["runs"], n=100))
    assert out["runs"] == len(b["runs"])
    assert round(out["trimmed_mean"], 4) == b["expected"]["trimmed_mean"]
    assert round(out["trimmed_passed"], 2) == b["expected"]["trimmed_passed"]


GROUPED = [t for t in CASES if "groups" in _fixture(t)]


@pytest.mark.parametrize("task", GROUPED)
def test_per_run_trimmed_by_group_reproduces_deepeval(task):
    """Per-category statistics: the same runs split by the sample's metadata field."""
    fx = _fixture(task)
    scores = []
    for group, g in fx["groups"].items():
        scores += _synthetic_scores(g["runs"], n=20, prefix=f"{group}-", metadata={"category": group})
    out = per_run_trimmed(group_by="category")(scores)
    for group, g in fx["groups"].items():
        assert round(out[f"{group}/trimmed_mean"], 4) == g["expected"]["trimmed_mean"]
        assert round(out[f"{group}/trimmed_passed"], 2) == g["expected"]["trimmed_passed"]
