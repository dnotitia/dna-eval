"""The request to the model under evaluation: one user turn holding
the sample input, no system prompt, max_tokens 16384, thinking via chat_template_kwargs.

Runs each task end to end against ``mockllm`` (no network for models; the dataset is fetched
from the Hub, so this needs read access to dnotitia/dna_*).
"""
import itertools

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from dna_eval.generation import MAX_TOKENS
from tests.parity.cases import CASES

JUDGE_JSON = '{"reason": "테스트", "score": 8}'


def _mock(text):
    return get_model("mockllm/model", custom_outputs=(ModelOutput.from_content("mockllm/model", text) for _ in itertools.count()))


@pytest.mark.network
@pytest.mark.parametrize("name", list(CASES))
def test_generation_request_matches_deepeval(name, tmp_path):
    case = CASES[name]
    log = inspect_eval(case["task"](epochs=2), model=_mock("모의 응답"), model_roles={"grader": _mock(JUDGE_JSON)},
                       limit=3, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success"
    for s in log.samples:
        # mockllm records no provider request, so check the ModelEvent itself: the messages
        # handed to the model and the GenerateConfig in force for that call.
        gen = next(ev for ev in s.events if ev.event == "model" and ev.role is None)
        assert [m.role for m in gen.input] == ["user"]
        assert gen.input[0].text == s.input
        assert gen.config.max_tokens == MAX_TOKENS
        assert gen.config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
        judge_calls = [ev for ev in s.events if ev.event == "model" and ev.role == "grader"]
        assert len(judge_calls) == case["judge_calls"]
        assert all(ev.config.temperature == 0.0 for ev in judge_calls)
        assert s.scores and all(sc.metadata.get("epoch") == s.epoch for sc in s.scores.values())
