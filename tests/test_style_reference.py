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
from dna_eval.solvers.style_reference import COLUMNS, NO_REFERENCE_ANSWER, ORIGIN_KEY, ReferenceCache, style_reference

SAMPLES = [Sample(id="s1", input="질문 하나"), Sample(id="s2", input="질문 둘")]


def _mock(*outputs, config=None):
    return get_model("mockllm/model", config=config,
                     custom_outputs=(o for o in itertools.cycle(outputs)))


def _reply(reasoning: str, answer: str) -> ModelOutput:
    return ModelOutput.from_message(ChatMessageAssistant(
        content=[ContentReasoning(reasoning=reasoning), ContentText(text=answer)], model="mockllm/model"))


def _run(tmp_path, reference_model=None, refresh=False, epochs=3, config=None):
    roles = {"grader": _mock(ModelOutput.from_content("mockllm/model", '{"score": 8, "reason": "ok"}'))}
    if reference_model is not None:
        roles["reference"] = reference_model
    task = Task(dataset=SAMPLES,
                solver=[style_reference("tiny-ref", "rev0", config=config or GenerateConfig(max_tokens=64),
                                        refresh=refresh),
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


def _reference_events(log):
    return [e for s in log.samples for e in s.events if e.event == "model" and e.role == "reference"]


@pytest.mark.parametrize("field,first_value,second_value", [
    ("temperature", 0.1, 0.9),
    ("max_tokens", 16, 32),
    ("top_p", 0.2, 0.8),
])
def test_effective_role_settings_split_cache(tmp_path, cache_dir, field, first_value, second_value):
    first = _run(tmp_path / "first", _mock(_reply("first", "first answer"),
                 config=GenerateConfig(**{field: first_value})), epochs=1)
    second = _run(tmp_path / "second", _mock(_reply("second", "second answer"),
                  config=GenerateConfig(**{field: second_value})), epochs=1)

    assert first.status == second.status == "success"
    assert _reference_calls(first) == _reference_calls(second) == len(SAMPLES)
    assert all(getattr(e.config, field) == first_value for e in _reference_events(first))
    assert all(getattr(e.config, field) == second_value for e in _reference_events(second))
    assert all(s.metadata["reference_content"] == "second answer" for s in second.samples)
    files = list(cache_dir.glob("style/rev0/tiny-ref.*.parquet"))
    assert len(files) == 2
    settings = [json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))["config"] for f in files]
    assert {s[field] for s in settings} == {first_value, second_value}


def test_same_effective_settings_reuse_cache_despite_task_defaults(tmp_path, cache_dir):
    role_config = GenerateConfig(temperature=0.4, max_tokens=32)
    first = _run(tmp_path / "first", _mock(_reply("first", "first answer"), config=role_config), epochs=1,
                 config=GenerateConfig(temperature=0.1, max_tokens=64))
    second = _run(tmp_path / "second", _mock(_reply("unused", "unused answer"), config=role_config), epochs=1,
                  config=GenerateConfig(temperature=0.9, max_tokens=128))

    assert first.status == second.status == "success"
    assert _reference_calls(first) == len(SAMPLES)
    assert all(e.config.temperature == 0.4 and e.config.max_tokens == 32 for e in _reference_events(first))
    assert _reference_calls(second) == 0
    assert all(s.store[ORIGIN_KEY] == "cache" and s.metadata["reference_content"] == "first answer"
               for s in second.samples)
    assert len(list(cache_dir.glob("style/rev0/tiny-ref.*.parquet"))) == 1
    assert role_config.temperature == 0.4 and role_config.max_tokens == 32


def test_runtime_only_role_settings_reuse_cache(tmp_path, cache_dir):
    first = _run(tmp_path / "first", _mock(_reply("first", "first answer"),
                 config=GenerateConfig(temperature=0.4, max_connections=1, max_retries=1, timeout=30)), epochs=1)
    second = _run(tmp_path / "second", _mock(_reply("unused", "unused answer"),
                  config=GenerateConfig(temperature=0.4, max_connections=4, max_retries=2, timeout=60)), epochs=1)

    assert first.status == second.status == "success"
    assert _reference_calls(first) == len(SAMPLES)
    assert _reference_calls(second) == 0
    assert all(s.metadata["reference_content"] == "first answer" for s in second.samples)
    files = list(cache_dir.glob("style/rev0/tiny-ref.*.parquet"))
    assert len(files) == 1
    settings = json.loads(files[0].with_suffix(".json").read_text(encoding="utf-8"))["config"]
    assert settings["temperature"] == 0.4
    assert not {"max_connections", "max_retries", "timeout"}.intersection(settings)


def test_existing_cache_without_role_overrides_is_reused(tmp_path, cache_dir):
    # Legacy key for model=mockllm/model and config={"max_tokens": 64}.
    path = cache_dir / "style/rev0/tiny-ref.e1051b2f4f7d.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"id": s.id, "input": s.input, "reference_reasoning": "legacy reasoning",
                   "reference_content": "legacy answer"} for s in SAMPLES], columns=COLUMNS).to_parquet(path, index=False)

    log = _run(tmp_path, _mock(_reply("unused", "unused answer")), epochs=1)

    assert log.status == "success"
    assert _reference_calls(log) == 0
    assert all(s.store[ORIGIN_KEY] == "cache" and s.metadata["reference_content"] == "legacy answer"
               for s in log.samples)
    assert list(cache_dir.glob("style/rev0/*.parquet")) == [path]


def test_nested_settings_order_is_stable_and_configs_are_not_mutated():
    first_config = GenerateConfig(max_tokens=64, extra_body={"top_k": 7, "chat_template_kwargs": {
        "enable_thinking": True, "preserve_special_tokens": False}})
    second_config = GenerateConfig(max_tokens=64, extra_body={"chat_template_kwargs": {
        "preserve_special_tokens": False, "enable_thinking": True}, "top_k": 7})
    before = first_config.model_dump()
    first_model = _mock(_reply("unused", "unused"), config=first_config)
    second_model = _mock(_reply("unused", "unused"), config=second_config)
    task_config = GenerateConfig(max_tokens=64)
    first = ReferenceCache("tiny-ref", "rev0", task_config, refresh=False)
    second = ReferenceCache("tiny-ref", "rev0", task_config, refresh=False)

    first.open(first_model)
    second.open(second_model)

    assert first.path == second.path
    assert first.settings["config"]["extra_body"] == first_config.extra_body
    assert first_config.model_dump() == first_model.config.model_dump() == before
    assert task_config.model_dump(exclude_none=True) == {"max_tokens": 64}
