"""The math benchmark tasks: prompts, pinned data, and one run end to end against ``mockllm``."""
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from dna_eval.tasks.benchmarks import aime2025 as aime2025_module
from dna_eval.tasks.benchmarks import aime2026 as aime2026_module
from dna_eval.tasks.benchmarks import hrm8k_ksm as ksm_module
from dna_eval.tasks.benchmarks.aime2025 import aime2025
from dna_eval.tasks.benchmarks.aime2026 import aime2026
from dna_eval.tasks.benchmarks.hrm8k_ksm import hrm8k_ksm

AIME = [(aime2025, aime2025_module, "MathArena/aime_2025"), (aime2026, aime2026_module, "MathArena/aime_2026")]


@pytest.mark.parametrize("task_fn, module, _", AIME)
def test_aime_prompt_is_the_instruction_then_the_problem(task_fn, module, _):
    sample = module.record_to_sample({"problem_idx": 3, "problem": "Find $x$.", "answer": 42})
    assert sample.id == 3 and sample.target == "42"
    assert sample.input == ("Put your final answer within \\boxed{}.\n"
                            "The answer is an integer between 0 and 999 inclusive.\n\nFind $x$.")


def test_ksm_prompt_is_the_lm_eval_prompt():
    record = {"question": "  1+1은?  ", "answer": "2", "category": "전공수학", "difficulty": "d"}
    sample = ksm_module.record_to_sample(record)
    assert sample.input == ("주어진 문제를 풀어보세요.\n"
                            "문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
                            "문제: 1+1은?\n답변:")
    assert sample.target == "2" and sample.metadata == {"category": "전공수학", "difficulty": "d"}


@pytest.mark.parametrize("module", [aime2025_module, aime2026_module, ksm_module])
def test_revision_is_a_pinned_commit(module):
    assert re.fullmatch(r"[0-9a-f]{40}", module.DATASET_REVISION)


@pytest.mark.network
@pytest.mark.parametrize("task_fn, module, hub_id", AIME)
def test_aime_data_at_the_pinned_revision(task_fn, module, hub_id):
    task = task_fn()
    assert task.epochs == 8
    samples = list(task.dataset)
    assert len(samples) == 30
    assert sorted(s.id for s in samples) == list(range(1, 31))
    assert all(re.fullmatch(r"\d{1,3}", s.target) and 0 <= int(s.target) <= 999 for s in samples)
    assert all(s.input.startswith("Put your final answer within \\boxed{}.\n") for s in samples)


@pytest.mark.network
def test_ksm_data_at_the_pinned_revision():
    task = hrm8k_ksm()
    assert task.epochs == 1
    samples = list(task.dataset)
    assert len(samples) == 1428
    assert all(s.target and s.metadata["category"] for s in samples)
    counts = {c: sum(1 for s in samples if s.metadata["category"] == c) for c in {s.metadata["category"] for s in samples}}
    assert counts == {"KMO": 730, "전공수학": 344, "수능/모의고사": 210, "KMS": 82, "KJMO": 62}
    assert sum(1 for s in samples if not re.fullmatch(r"-?\d+(\.\d+)?", s.target)) == 222   # LaTeX answers
    assert all(s.input.startswith("주어진 문제를 풀어보세요.") and s.input.endswith("\n답변:") for s in samples)


def _mock(text, stop_reason="stop"):
    out = ModelOutput.from_content("mockllm/model", text)
    out.choices[0].stop_reason = stop_reason
    return get_model("mockllm/model", custom_outputs=[out] * 200)


@pytest.mark.network
@pytest.mark.parametrize("task_fn, module, _", AIME)
def test_aime_runs_with_the_benchmark_request(task_fn, module, _, tmp_path):
    log = inspect_eval(task_fn(epochs=2), model=_mock("no idea"), limit=3, display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success"
    for s in log.samples:
        gen = next(ev for ev in s.events if ev.event == "model")
        assert [m.role for m in gen.input] == ["user"]     # one user turn (long text is stored as an attachment)
        assert gen.config.max_tokens is None
        assert gen.config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    metrics = {k: m.value for sc in log.results.scores for k, m in sc.metrics.items()}
    assert metrics["accuracy"] == 0.0 and metrics["format_failure"] == 1.0


@pytest.mark.network
def test_ksm_reports_accuracy_by_category_and_diagnostics(tmp_path):
    log = inspect_eval(hrm8k_ksm(thinking=False), model=_mock(r"$\boxed{1}$"), limit=20, display="none",
                       log_dir=str(tmp_path))[0]
    assert log.status == "success"
    gen = next(ev for ev in log.samples[0].events if ev.event == "model")
    assert gen.config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    names = {k for sc in log.results.scores for k in sc.metrics}
    assert {"accuracy", "stderr", "truncated", "format_failure", "accuracy_excl_truncated"} <= names
    assert any(n not in {"accuracy", "stderr", "truncated", "format_failure", "accuracy_excl_truncated"} for n in names), \
        "expected per-category accuracy entries"
