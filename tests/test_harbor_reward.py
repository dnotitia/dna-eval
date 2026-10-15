"""``harbor_reward``: the verifier's reward as a score, with a fake inner scorer (no sandbox).

``inspect_harbor.harbor_scorer`` copies the tests into a container and runs them; here it is
replaced by a function that returns what it would return, so the wrapper's own behaviour is
checked: the reward as the value, the agent solver's keys copied into the metadata, a negative
reward turned into an error, and the verifier's own errors left to propagate.
"""
import asyncio

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState, solver

from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS
from dna_eval.scorers import harbor_reward as module
from dna_eval.scorers.harbor_reward import DATASET, SCAFFOLD, harbor_reward

DATASET_ID = "org/name@sha256:abc"


def _state(store=None):
    state = TaskState(model="mockllm/model", sample_id=1, epoch=1, input="fix it", messages=[])
    for key, value in (store or {}).items():
        state.store.set(key, value)
    return state


def _fake_inner(monkeypatch, result):
    """Make the wrapped ``harbor_scorer`` return ``result`` (a Score) or raise it (an exception)."""
    async def inner(state, target):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module, "_inner_scorer", lambda timeout: inner)


def _score(state=None, **kwargs):
    scorer = harbor_reward(dataset=DATASET_ID, scaffold="mini_swe_agent", **kwargs)
    return asyncio.run(scorer(state or _state(), Target("")))


def test_passing_reward_is_one_and_metadata_is_filled(monkeypatch):
    _fake_inner(monkeypatch, Score(value=1.0, answer="PASS", explanation="Test exit code: 0"))
    score = _score(_state({LIMIT_REACHED: "time", TURNS: 17}))
    assert score.value == 1.0 and score.answer == "PASS" and score.explanation == "Test exit code: 0"
    assert score.metadata == {LIMIT_REACHED: "time", TURNS: 17, DATASET: DATASET_ID, SCAFFOLD: "mini_swe_agent"}


def test_failing_reward_is_a_zero_not_an_error(monkeypatch):
    _fake_inner(monkeypatch, Score(value=0.0, answer="FAIL"))
    score = _score()
    assert score.value == 0.0 and score.answer == "FAIL"
    assert score.metadata[LIMIT_REACHED] is None and score.metadata[TURNS] is None   # the solver left nothing


def test_reward_dict_is_kept(monkeypatch):
    _fake_inner(monkeypatch, Score(value=1.0, answer="PASS", metadata={"reward_dict": {"reward": 1, "pass_fraction": 0.9}}))
    assert _score().metadata["reward_dict"] == {"reward": 1, "pass_fraction": 0.9}


@pytest.mark.parametrize("reward", [-1.0, -0.5])
def test_negative_reward_is_the_verifiers_crash_marker_and_raises(monkeypatch, reward):
    _fake_inner(monkeypatch, Score(value=reward, answer="FAIL", explanation="grader crashed"))
    with pytest.raises(RuntimeError, match="crash.*grader crashed"):
        _score()


@pytest.mark.parametrize("error", [TimeoutError("verifier timed out"), FileNotFoundError("no reward file"),
                                   ValueError("unparsable reward")])
def test_verifier_errors_propagate(monkeypatch, error):
    _fake_inner(monkeypatch, error)
    with pytest.raises(type(error), match=str(error)):
        _score()


def test_timeout_is_passed_to_the_wrapped_scorer(monkeypatch):
    seen = []

    async def inner(state, target):
        return Score(value=1.0)

    monkeypatch.setattr(module, "_inner_scorer", lambda timeout: seen.append(timeout) or inner)
    harbor_reward(default_verifier_timeout_sec=123)
    assert seen == [123]


def test_missing_dependency_names_the_install_command(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "inspect_harbor":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError, match=r'pip install "dna-eval\[agentic\]"'):
        module._inner_scorer(600)


def test_metrics_in_a_run(monkeypatch, tmp_path):
    """Through ``inspect eval``: the headline is accuracy, with its stderr and ``agent_diagnostics``;
    a run ended by a limit is counted; a crashed verifier is a sample error, not a score."""
    async def inner(state, target):
        reward = {"pass": 1.0, "fail": 0.0, "crash": -1.0}[state.input_text]
        return Score(value=reward, answer="PASS" if reward > 0 else "FAIL")

    monkeypatch.setattr(module, "_inner_scorer", lambda timeout: inner)

    @solver
    def stub_agent():
        async def solve(state, generate):
            state.store.set(TURNS, 10 if state.input_text == "pass" else 30)
            state.store.set(LIMIT_REACHED, "turns" if state.input_text == "fail" else None)
            return state
        return solve

    task = Task(dataset=[Sample(id=i, input=i) for i in ("pass", "fail", "crash")], solver=stub_agent(),
                scorer=harbor_reward(dataset=DATASET_ID, scaffold="react"))
    log = inspect_eval(task, model=get_model("mockllm/model"), display="none", log_dir=str(tmp_path),
                       fail_on_error=False)[0]
    metrics = {k: v.value for s in log.results.scores for k, v in s.metrics.items()}
    assert metrics["accuracy"] == pytest.approx(0.5)            # the crash is out of the denominator
    assert {"stderr", "limit_reached", "mean_turns"} <= set(metrics)
    assert metrics["limit_reached"] == pytest.approx(0.5) and metrics["mean_turns"] == pytest.approx(20)
    by_id = {s.id: s for s in log.samples}
    assert by_id["crash"].error is not None and "crash" in by_id["crash"].error.message
    assert by_id["pass"].scores["harbor_reward"].metadata[SCAFFOLD] == "react"
