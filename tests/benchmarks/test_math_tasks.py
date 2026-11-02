"""The HRM8K-KSM task: prompt, pinned data, and one run end to end against ``mockllm``."""
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from dna_eval.tasks.benchmarks import hrm8k_ksm as ksm_module
from dna_eval.tasks.benchmarks.hrm8k_ksm import hrm8k_ksm


def test_ksm_prompt_is_the_lm_eval_prompt():
    record = {"question": "  1+1은?  ", "answer": "2", "category": "전공수학", "difficulty": "d"}
    sample = ksm_module.record_to_sample(record)
    assert sample.input == ("주어진 문제를 풀어보세요.\n"
                            "문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
                            "문제: 1+1은?\n답변:")
    assert sample.target == "2" and sample.metadata == {"category": "전공수학", "difficulty": "d"}


def test_revision_is_a_pinned_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", ksm_module.DATASET_REVISION)


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
