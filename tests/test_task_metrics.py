"""Task-level ``metrics`` must not hide the scorer's failure metrics."""
import itertools

from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model

from dna_eval.tasks import hallucination as hallucination_module


def _mock(text):
    return get_model("mockllm/model", custom_outputs=(ModelOutput.from_content("mockllm/model", text)
                                                      for _ in itertools.count()))


def test_hallucination_reports_failure_metrics(tmp_path, monkeypatch):
    samples = [Sample(id=i, input="q", metadata={"category": c, "pair_id": i, "difficulty": "easy", "mode": "m"})
               for i, c in enumerate(["false_premise", "grounding"])]
    monkeypatch.setattr(hallucination_module, "dna_dataset", lambda *a, **k: samples)
    task = hallucination_module.hallucination(epochs=2)
    log = inspect_eval(task, model=_mock("answer"), model_roles={"grader": _mock('{"score": 8, "reason": "ok"}')},
                       display="none", log_dir=str(tmp_path))[0]
    names = {k for s in log.results.scores for k in s.metrics}
    assert {"answer_failed", "answer_empty", "judge_failed"} <= names
    assert any(n.startswith("grounding/") for n in names)
