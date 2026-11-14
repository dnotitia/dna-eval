"""Inspect limit errors propagate out of ``call_with_retries`` instead of being retried, and the
sample is recorded and scored exactly as a call that kept failing would be."""
import asyncio
import itertools

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelName, ModelOutput, get_model
from inspect_ai.solver import TaskState
from inspect_ai.util import LimitExceededError

from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import EMPTY_ANSWER, EMPTY_REASONING, NO_ANSWER_KEY, REASONING_KEY, dna_generate
from dna_eval.solvers.style_reference import NO_REFERENCE_ANSWER, NO_REFERENCE_REASONING, style_reference
from dna_eval.utils import retry


def _mock(text):
    return get_model("mockllm/model", custom_outputs=(ModelOutput.from_content("mockllm/model", text)
                                                      for _ in itertools.count()))


def _judge_prompts(sample):
    """Prompts sent to the judge; long ones are stored as attachment references."""
    prompts = []
    for e in sample.events:
        if e.event == "model" and e.role == "grader":
            prompt = e.input[0].content
            prompts.append(sample.attachments.get(prompt.removeprefix("attachment://"), prompt))
    return prompts


def test_token_limit_is_recorded_and_scored_like_a_failed_call(tmp_path):
    task = Task(dataset=[Sample(id="s1", input="question")], solver=dna_generate(),
                scorer=rubric_scorer("performance"), token_limit=1)
    log = inspect_eval(task, model=_mock("a fairly long answer to the question"),
                       model_roles={"grader": _mock('{"score": 8, "reason": "ok"}')},
                       display="none", log_dir=str(tmp_path))[0]
    sample = log.samples[0]
    assert sample.limit is not None and sample.limit.type == "token"
    assert sample.store[REASONING_KEY] == EMPTY_REASONING
    assert sample.output.completion == EMPTY_ANSWER
    assert EMPTY_ANSWER in _judge_prompts(sample)[0]
    assert sample.scores
    for score in sample.scores.values():
        assert (score.metadata or {}).get(NO_ANSWER_KEY) == "failed"


def _state() -> TaskState:
    return TaskState(model=ModelName("mockllm/model"), sample_id="s1", epoch=1, input="question",
                     messages=[ChatMessageUser(content="question")])


def test_limit_after_an_empty_reply_is_recorded_as_empty(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", fake_sleep)
    calls = 0

    async def generate(state):
        nonlocal calls
        calls += 1
        if calls == 1:
            # neither content nor reasoning, so it is retried (a reasoning-only reply is not)
            state.output = ModelOutput.from_content("mockllm/model", "")
            return state
        raise LimitExceededError("token", value=10, limit=5)

    state = _state()
    with pytest.raises(LimitExceededError):
        asyncio.run(dna_generate()(state, generate))
    assert calls == 2 and sleeps == [1]          # one retry after the empty reply, none after the limit
    assert state.store.get(NO_ANSWER_KEY) == "empty"
    assert state.output.completion == EMPTY_ANSWER
    assert state.store.get(REASONING_KEY) == EMPTY_REASONING


def test_limit_on_the_first_call_is_recorded_as_failed():
    async def generate(state):
        raise LimitExceededError("message", value=1, limit=1)

    state = _state()
    with pytest.raises(LimitExceededError):
        asyncio.run(dna_generate()(state, generate))
    assert state.store.get(NO_ANSWER_KEY) == "failed"
    assert state.output.completion == EMPTY_ANSWER
    assert state.store.get(REASONING_KEY) == EMPTY_REASONING


def test_limit_during_reference_generation_leaves_placeholders(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("DNA_EVAL_CACHE_DIR", str(cache_dir))
    task = Task(dataset=[Sample(id="s1", input="question")],
                solver=[style_reference("tiny-ref", "rev0", config=GenerateConfig(max_tokens=64)), dna_generate()],
                scorer=[rubric_scorer("reasoning_style"), rubric_scorer("content_style")], token_limit=1)
    log = inspect_eval(task, model=_mock("a fairly long answer"),
                       model_roles={"grader": _mock('{"score": 8, "reason": "ok"}'),
                                    "reference": _mock("a fairly long reference answer")},
                       display="none", log_dir=str(tmp_path / "logs"))[0]
    sample = log.samples[0]
    assert sample.limit is not None and sample.limit.type == "token"
    prompts = _judge_prompts(sample)
    assert len(prompts) == 2
    assert any(EMPTY_REASONING in p and NO_REFERENCE_REASONING in p for p in prompts)   # reasoning_style
    assert any(EMPTY_ANSWER in p and NO_REFERENCE_ANSWER in p for p in prompts)         # content_style
    assert not list(cache_dir.glob("style/rev0/*.parquet"))
