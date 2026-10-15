"""The SciCode task: pinned data, wiring, and one run end to end against ``mockllm`` and the local sandbox."""
import re
import sys
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.model import ModelOutput, get_model

from dna_eval.scorers.scicode_tests import scicode_tests
from dna_eval.solvers.scicode_steps import scicode_steps
from dna_eval.tasks.benchmarks import benchmark_generate_config
from dna_eval.tasks.benchmarks import scicode as scicode_module
from dna_eval.tasks.benchmarks.scicode import (DATASET_REVISION, H5_SHA256, H5_URL, SANDBOX, SKIPPED, STEPS_KEY,
                                               record_to_sample, scicode)


def test_pins_are_a_commit_and_a_checksum():
    assert re.fullmatch(r"[0-9a-f]{40}", DATASET_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", H5_SHA256)
    assert re.search(r"/resolve/[0-9a-f]{40}/test_data\.h5$", H5_URL)


def test_record_to_sample_keeps_what_the_prompts_and_tests_need():
    record = {"problem_id": "62", "problem_name": "dmrg", "problem_description_main": "Do DMRG.",
              "required_dependencies": "import numpy as np", "general_solution": "secret",
              "sub_steps": [{"step_number": f"62.{i}", "step_description_prompt": f"d{i}", "step_background": f"b{i}",
                             "function_header": f"def f{i}():", "return_line": "return 1", "test_cases": [f"t{i}"],
                             "ground_truth_code": "SOLUTION"} for i in (1, 2)]}
    sample = record_to_sample(record)
    assert sample.id == "62" and sample.input == "Do DMRG."
    assert sample.metadata["required_dependencies"] == "import numpy as np"
    first, second = sample.metadata["sub_steps"]
    assert first["skipped_code"] and "class Block" in first["skipped_code"]      # 62.1 is skipped
    assert second["skipped_code"] is None
    assert set(second) == {"step_number", "step_description_prompt", "step_background", "function_header",
                           "return_line", "test_cases", "skipped_code"}
    assert "SOLUTION" not in repr(sample.metadata)


def test_the_three_skipped_steps_have_gold_code_files():
    vendor = Path(scicode_module.vendored.__file__).parent
    assert set(SKIPPED) == {"13.6", "62.1", "76.3"}
    assert all((vendor / name).is_file() for name in SKIPPED.values())


def test_the_sandbox_has_no_network_and_a_pinned_image():
    kind, compose = SANDBOX
    assert kind == "docker"
    text = Path(compose).read_text()
    assert "network_mode: none" in text and "default:" in text
    dockerfile = (Path(compose).parent / "Dockerfile").read_text()
    assert "python:3.11" in dockerfile and all(p in dockerfile for p in ("numpy==", "scipy==", "sympy==", "h5py==", "matplotlib=="))


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """The task with its dataset and test file replaced by local stand-ins."""
    pytest.importorskip("h5py")
    monkeypatch.setattr(scicode_module, "hf_dataset", lambda *a, **k: MemoryDataset([Sample(id="p", input="x")]))
    monkeypatch.setattr(scicode_module, "h5_file", lambda: tmp_path / "unused.h5")


def test_task_wiring(offline):
    task = scicode()
    assert task.epochs == 3
    assert task.sandbox.type == "docker" and Path(task.sandbox.config).is_file()
    assert task.config.max_tokens is None
    assert task.config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert task.headline_metric.metric == "accuracy"
    assert scicode(thinking=False, epochs=1).config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_task_requires_h5py(monkeypatch):
    monkeypatch.setattr(scicode_module, "hf_dataset", lambda *a, **k: MemoryDataset([Sample(id="p", input="x")]))
    monkeypatch.setattr(scicode_module, "h5_file", lambda: pytest.fail("the download starts before h5py is checked"))
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(RuntimeError, match="pip install"):
        scicode()


# One problem of two sub-steps, tested against targets built here.
def _problem() -> Sample:
    steps = [
        {"step_number": "S.1", "step_description_prompt": "Square a number.", "step_background": "BG1",
         "function_header": "def square(x):\n    '''Square x.'''", "return_line": "    return y",
         "test_cases": ["assert np.allclose(square(3.0), target)"], "skipped_code": None},
        {"step_number": "S.2", "step_description_prompt": "Add one to the square.", "step_background": "BG2",
         "function_header": "def square_plus_one(x):\n    '''Square x and add one.'''", "return_line": "    return y",
         "test_cases": ["assert np.allclose(square_plus_one(3.0), target)"], "skipped_code": None}]
    return Sample(id="S", input="Squares.", target="", metadata={"required_dependencies": "import numpy as np",
                                                                 "sub_steps": steps})


REPLIES = ["```python\nimport numpy as np\ndef square(x):\n    return x * x\n```",
           "```python\ndef square_plus_one(x):\n    return square(x) + 1\n```"]


def test_end_to_end_with_mockllm_and_local_sandbox(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    pytest.importorskip("sympy")
    with h5py.File(tmp_path / "t.h5", "w") as f:
        f["S.1/test1/var1"] = 9.0
        f["S.2/test1/var1"] = 10.0
    outputs = [ModelOutput.from_content("mockllm/model", r) for r in REPLIES]
    task = Task(dataset=[_problem()], solver=scicode_steps(key=STEPS_KEY),
                scorer=scicode_tests(tmp_path / "t.h5", key=STEPS_KEY, python=sys.executable),
                epochs=1, config=benchmark_generate_config(True), sandbox="local")
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=outputs), display="none",
                       log_dir=str(tmp_path / "logs"))[0]
    assert log.status == "success", log.error
    metrics = {m: v.value for s in log.results.scores for m, v in s.metrics.items()}
    assert metrics["accuracy"] == 1.0 and metrics["sub_step_accuracy"] == 1.0
    assert metrics["truncated"] == 0.0 and metrics["format_failure"] == 0.0
    for sample in log.samples:
        calls = [ev for ev in sample.events if ev.event == "model"]
        assert len(calls) == 2                                              # one request per sub-step
        assert all([m.role for m in c.input] == ["user"] for c in calls)    # each a fresh one-turn prompt
        assert all(c.config.max_tokens is None for c in calls)
        assert sample.store[STEPS_KEY][1]["previous_code"].count("def square(x):") == 1   # step 1's code is its context
        assert sample.scores["scicode_tests"].metadata["steps"] == [
            {"step_number": "S.1", "status": "pass"}, {"step_number": "S.2", "status": "pass"}]


@pytest.mark.network
def test_dataset_at_the_pinned_revision():
    samples = list(hf_dataset("SciCode1/SciCode", split="test", revision=DATASET_REVISION,
                              sample_fields=record_to_sample))
    assert len(samples) == 65 and len({s.id for s in samples}) == 65
    steps = [st for s in samples for st in s.metadata["sub_steps"]]
    assert len(steps) == 291
    skipped = sorted(st["step_number"] for st in steps if st["skipped_code"] is not None)
    assert skipped == ["13.6", "62.1", "76.3"]
    assert sum(1 for st in steps if st["skipped_code"] is None) == 288
    assert all(st["test_cases"] for st in steps if st["skipped_code"] is None)
    assert sum(len(st["test_cases"]) for st in steps) == 915
    assert all(s.metadata["required_dependencies"] and not s.metadata["sub_steps"][0].get("ground_truth_code")
               for s in samples)


@pytest.mark.network
def test_test_file_mirror_serves_the_pinned_checksum():
    from huggingface_hub import get_hf_file_metadata
    meta = get_hf_file_metadata(H5_URL)
    assert meta.size == 1_049_345_865
    assert meta.etag == H5_SHA256       # for a Git LFS file the Hub's ETag is the SHA-256 of the content
