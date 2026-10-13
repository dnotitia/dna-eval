"""Locally generated style references: generated once per sample, cached, reused, refreshable."""
import itertools
import json

import pandas as pd
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, GenerateConfig, ModelOutput, get_model

from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate
from dna_eval.solvers.style_reference import COLUMNS, NO_REFERENCE_ANSWER, ORIGIN_KEY, style_reference

SAMPLES = [Sample(id="s1", input="질문 하나"), Sample(id="s2", input="질문 둘")]


def _mock(*outputs):
    return get_model("mockllm/model", custom_outputs=(o for o in itertools.cycle(outputs)))


def _reply(reasoning: str, answer: str) -> ModelOutput:
    return ModelOutput.from_message(ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning), ContentText(text=answer)], model="mockllm/model"))


def _run(tmp_path, reference_model=None, refresh=False, epochs=3):
    roles = {"grader": _mock(ModelOutput.from_content("mockllm/model", '{"score": 8, "reason": "ok"}'))}
    if reference_model is not None:
        roles["reference"] = reference_model
    task = Task(dataset=SAMPLES,
                solver=[style_reference("tiny-ref", "rev0", config=GenerateConfig(max_tokens=64), refresh=refresh),
                        dna_generate()],
                scorer=rubric_scorer("content_style"), epochs=epochs)
    return inspect_eval(task, model=_mock(_reply("모델 생각", "모델 답")), model_roles=roles,
                        display="none", log_dir=str(tmp_path / "logs"))[0]


def _reference_calls(log):
    return sum(1 for s in log.samples for e in s.events if e.event == "model" and e.role == "reference")


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DNA_EVAL_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_generated_once_per_sample_and_cached_in_hub_schema(tmp_path, cache_dir):
    log = _run(tmp_path, _mock(_reply("  참조 생각 ", " 참조 답 ")))
    assert log.status == "success"
    assert _reference_calls(log) == len(SAMPLES)                      # not once per epoch
    assert {s.store[ORIGIN_KEY] for s in log.samples} <= {"generated", "cache"}
    assert all(s.metadata["reference_content"] == "참조 답" for s in log.samples)
    files = list(cache_dir.glob("style/rev0/tiny-ref.*.parquet"))
    assert len(files) == 1
    table = pd.read_parquet(files[0])
    assert list(table.columns) == COLUMNS and len(table) == len(SAMPLES)
    assert table["reference_reasoning"].tolist() == ["참조 생각", "참조 생각"]
    provenance = json.loads(files[0].with_suffix(".json").read_text(encoding="utf-8"))
    assert provenance["reference"] == "tiny-ref" and provenance["config"]["max_tokens"] == 64


def test_second_run_reads_the_cache(tmp_path):
    _run(tmp_path, _mock(_reply("참조 생각", "참조 답")))
    log = _run(tmp_path, _mock(_reply("다른 생각", "다른 답")))
    assert _reference_calls(log) == 0
    assert all(s.store[ORIGIN_KEY] == "cache" and s.metadata["reference_content"] == "참조 답" for s in log.samples)


def test_refresh_regenerates(tmp_path):
    _run(tmp_path, _mock(_reply("참조 생각", "참조 답")))
    log = _run(tmp_path, _mock(_reply("새 생각", "새 답")), refresh=True)
    assert _reference_calls(log) == len(SAMPLES)
    assert all(s.metadata["reference_content"] == "새 답" for s in log.samples)


def test_missing_reference_role_fails_loudly(tmp_path):
    log = _run(tmp_path, reference_model=None, epochs=1)
    assert log.status == "error" and "reference" in log.error.message


def test_failed_reference_is_judged_as_placeholder_and_not_cached(tmp_path, cache_dir):
    empty = ModelOutput.from_content("mockllm/model", "")
    log = _run(tmp_path, _mock(empty), epochs=1)
    assert all(s.metadata["reference_content"] == NO_REFERENCE_ANSWER for s in log.samples)
    assert not list(cache_dir.glob("style/rev0/*.parquet"))
