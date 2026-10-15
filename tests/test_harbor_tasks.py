"""``terminal_bench``, ``swe_bench`` and ``deep_swe``: task construction and the agent solver.

No container, no model server and no download (except the ``network`` tests, which read the
dataset lists from the Harbor hub): ``inspect_harbor.harbor`` is replaced by a function that
returns a one-sample task, the agent by small stand-ins run against ``mockllm``. What is checked
is what dna-eval adds: the pinned digests, the scaffold and limit defaults, the warning for
Terminal-Bench 4.0, the per-sample time limit and turn limit, and what is recorded in the log.
"""
import asyncio
import logging
import re

import pytest
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import registry_info, registry_params
from inspect_ai.agent import Agent, agent
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, get_model
from inspect_ai.scorer import Score
from inspect_ai.solver import TaskState

pytest.importorskip("inspect_harbor")
pytest.importorskip("inspect_swe")

import inspect_harbor  # noqa: E402

from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS  # noqa: E402
from dna_eval.scorers import harbor_reward as reward_module  # noqa: E402
from dna_eval.solvers import agent_scaffold  # noqa: E402
from dna_eval.tasks.agentic import deep_swe as deep_swe_module  # noqa: E402
from dna_eval.tasks.agentic import swe_bench as swe_bench_module  # noqa: E402
from dna_eval.tasks.agentic import terminal_bench as terminal_bench_module  # noqa: E402
from dna_eval.tasks.agentic.deep_swe import deep_swe  # noqa: E402
from dna_eval.tasks.agentic.swe_bench import swe_bench  # noqa: E402
from dna_eval.tasks.agentic.terminal_bench import terminal_bench  # noqa: E402

DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@pytest.fixture
def harbor_calls(monkeypatch):
    """``inspect_harbor.harbor`` returns a one-sample task and records its arguments."""
    calls = []

    def fake_harbor(**kwargs):
        calls.append(kwargs)
        return Task(dataset=[Sample(id="t1", input="solve it", metadata={"harbor_config": {"agent": {"timeout_sec": 900}}})])

    monkeypatch.setattr(inspect_harbor, "harbor", fake_harbor)
    return calls


def _solver_args(task):
    return registry_params(task.solver)


# --- pinned data -------------------------------------------------------------------------------

def test_every_dataset_is_pinned_by_a_distinct_digest():
    digests = [terminal_bench_module.DATASET_DIGEST, terminal_bench_module.DATASET_DIGEST_4,
               deep_swe_module.DATASET_DIGEST, *(d for _, d in swe_bench_module.VARIANTS.values())]
    assert all(DIGEST.match(d) for d in digests)
    assert len(set(digests)) == len(digests)


def test_terminal_bench_2_1_is_the_default_and_pinned(harbor_calls):
    task = terminal_bench()
    assert harbor_calls == [{"package_name": "terminal-bench/terminal-bench-2-1",
                             "package_ref": terminal_bench_module.DATASET_DIGEST, "dataset_task_names": None,
                             "n_tasks": None, "sandbox_env_name": "docker"}]
    assert task.metadata["dataset"] == f"terminal-bench/terminal-bench-2-1@{terminal_bench_module.DATASET_DIGEST}"
    assert task.metadata["version"] == "2.1" and "experimental" not in task.metadata
    assert (task.epochs, task.metadata["max_steps"]) == (5, 0)


def test_terminal_bench_4_is_experimental_and_says_so(harbor_calls, caplog):
    with caplog.at_level(logging.WARNING, logger=terminal_bench_module.__name__):
        task = terminal_bench(version="4.0")
    assert harbor_calls[0]["package_name"] == "terminal-bench/terminal-bench"
    assert harbor_calls[0]["package_ref"] == terminal_bench_module.DATASET_DIGEST_4
    assert any("experimental" in r.message and "separate verifier" in r.message for r in caplog.records)
    assert "experimental" in task.metadata and task.metadata["max_steps"] == 500
    assert "experimental" in terminal_bench_module.__doc__.lower()


def test_terminal_bench_2_1_logs_no_warning(harbor_calls, caplog):
    with caplog.at_level(logging.WARNING, logger=terminal_bench_module.__name__):
        terminal_bench()
    assert not caplog.records


def test_terminal_bench_max_steps_zero_is_kept_not_replaced_by_the_default(harbor_calls):
    assert terminal_bench(version="4.0", max_steps=0).metadata["max_steps"] == 0
    assert terminal_bench(max_steps=40).metadata["max_steps"] == 40


def test_unknown_versions_and_variants_are_rejected(harbor_calls):
    with pytest.raises(ValueError, match="2.1, 4.0"):
        terminal_bench(version="3.0")
    with pytest.raises(ValueError, match="pro, pro_hardened, verified"):
        swe_bench(variant="lite")
    assert harbor_calls == []


@pytest.mark.parametrize("variant, package", [("pro", "scale-ai/swe-bench-pro"), ("pro_hardened", "cais/swebenchpro"),
                                              ("verified", "swe-bench/swe-bench-verified")])
def test_swe_bench_variants(harbor_calls, variant, package):
    task = swe_bench(variant=variant)
    assert harbor_calls[0]["package_name"] == package
    assert harbor_calls[0]["package_ref"] == swe_bench_module.VARIANTS[variant][1]
    assert task.metadata["variant"] == variant
    assert (task.epochs, task.metadata["max_steps"], task.metadata["command_timeout"]) == (1, 250, 60)


def test_swe_bench_defaults_to_pro(harbor_calls):
    assert swe_bench().metadata["variant"] == "pro"
    assert harbor_calls[0]["package_name"] == "scale-ai/swe-bench-pro"


def test_swe_bench_limits_can_be_lifted(harbor_calls):
    task = swe_bench(max_steps=0, command_timeout=300)
    assert (task.metadata["max_steps"], task.metadata["command_timeout"]) == (0, 300)


def test_deep_swe_follows_the_leaderboard_setup(harbor_calls):
    task = deep_swe()
    assert harbor_calls[0]["package_name"] == "datacurve/deep-swe-1-1"
    assert harbor_calls[0]["package_ref"] == deep_swe_module.DATASET_DIGEST
    assert (task.epochs, task.metadata["max_steps"], task.metadata["command_timeout"]) == (1, 0, None)
    assert task.metadata["scaffold"] == "mini_swe_agent"


# --- what a task is made of --------------------------------------------------------------------

def test_task_parts(harbor_calls):
    task = terminal_bench(scaffold="react", thinking=False, sandbox="k8s", task_names="terminal-bench/x*", n_tasks=3)
    assert harbor_calls[0]["sandbox_env_name"] == "k8s"
    assert harbor_calls[0]["dataset_task_names"] == "terminal-bench/x*" and harbor_calls[0]["n_tasks"] == 3
    assert registry_info(task.solver).name == "agent_solver"
    assert _solver_args(task)["scaffold"] == "react"
    assert registry_info(task.scorer[0]).name.endswith("harbor_reward")
    assert registry_params(task.scorer[0])["scaffold"] == "react"
    assert task.metadata["scaffold"] == "react"
    assert task.config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert task.config.max_tokens is None
    assert [r.__name__ for r in task.epochs_reducer] == ["mean_score"] or "mean" in str(task.epochs_reducer)
    assert task.headline_metric.metric == "accuracy"
    assert set(task.metadata["packages"]) == {"inspect_harbor", "inspect_swe"}


def test_unknown_scaffold_is_rejected_before_anything_is_loaded(harbor_calls):
    with pytest.raises(ValueError, match="mini_swe_agent, react, claude_code, codex_cli"):
        deep_swe(scaffold="terminus")
    assert harbor_calls == []


def test_missing_dependency_names_the_install_command(harbor_calls, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "inspect_harbor" else real(name, *a))
    with pytest.raises(RuntimeError, match=r'pip install "dna-eval\[agentic\]"'):
        deep_swe()


# --- the mini-swe-agent configuration ----------------------------------------------------------

def test_mini_config_is_the_built_in_one_with_only_the_limits_changed():
    built_in = yaml.safe_load(agent_scaffold.MINI_SWE_AGENT_YAML)
    assert built_in["agent"]["step_limit"] == 0 and built_in["agent"]["cost_limit"] == 3.0
    config = yaml.safe_load(agent_scaffold.mini_swe_agent_config(250, 60))
    assert (config["agent"]["step_limit"], config["agent"]["cost_limit"], config["environment"]["timeout"]) == (250, 0, 60)
    config["agent"].update(step_limit=0, cost_limit=3.0)
    config["environment"].pop("timeout")
    assert config == built_in                       # prompts and templates are untouched
    assert "timeout" not in yaml.safe_load(agent_scaffold.mini_swe_agent_config())["environment"]


# --- setting up the agent ----------------------------------------------------------------------

class _FakeSandbox:
    def __init__(self):
        self.files = {}

    async def write_file(self, path, contents):
        self.files[path] = contents


def _state(**metadata):
    return TaskState(model="mockllm/model", sample_id=1, epoch=1, input="x", messages=[], metadata=metadata)


def test_mini_swe_agent_setup(monkeypatch):
    box = _FakeSandbox()
    monkeypatch.setattr(agent_scaffold, "sandbox", lambda: box)
    built = asyncio.run(agent_scaffold._setup_agent(_state(agent_user="dev"), "mini_swe_agent", 250, 60))
    params = registry_params(built)
    assert registry_info(built).name.endswith("mini_swe_agent")
    assert params["user"] == "dev" and params["version"] == agent_scaffold.MINI_SWE_AGENT_VERSION
    assert params["env"]["MSWEA_COST_TRACKING"] == "ignore_errors"
    assert params["env"]["MSWEA_MINI_CONFIG_PATH"] == agent_scaffold.MINI_SWE_AGENT_CONFIG_PATH
    assert box.files == {agent_scaffold.MINI_SWE_AGENT_CONFIG_PATH: agent_scaffold.mini_swe_agent_config(250, 60)}


@pytest.mark.parametrize("scaffold", ["react", "claude_code", "codex_cli"])
def test_other_scaffolds_build(scaffold):
    built = asyncio.run(agent_scaffold._setup_agent(_state(agent_user=None), scaffold, 0, None))
    assert callable(built)


def test_agent_solver_rejects_an_unknown_scaffold():
    with pytest.raises(ValueError, match="unknown scaffold"):
        agent_scaffold.agent_solver("nope")


# --- limits, end to end against mockllm --------------------------------------------------------

def _stand_in(monkeypatch, behaviour):
    """Make ``agent_solver`` run ``behaviour(state)`` as the agent."""
    @agent
    def stand_in() -> Agent:
        async def execute(state):
            await behaviour(state)
            return state
        return execute

    async def setup(state, scaffold, step_limit, command_timeout):
        return stand_in()

    monkeypatch.setattr(agent_scaffold, "_setup_agent", setup)


async def _generate_forever(state):
    model = get_model("mockllm/model")
    while True:
        state.output = await model.generate(state.messages)
        state.messages.append(state.output.message)


def _run(tmp_path, monkeypatch, *, timeout_sec=None, max_steps=0, scaffold="react", reward=0.0):
    async def inner(state, target):
        return Score(value=reward, answer="PASS" if reward else "FAIL")

    monkeypatch.setattr(reward_module, "_inner_scorer", lambda timeout: inner)
    metadata = {"harbor_config": {"agent": {"timeout_sec": timeout_sec}}}
    task = Task(dataset=[Sample(id="s", input="go", metadata=metadata)], solver=agent_scaffold.agent_solver(scaffold, max_steps),
                scorer=reward_module.harbor_reward(dataset="org/n@sha256:0", scaffold=scaffold))
    log = inspect_eval(task, model=get_model("mockllm/model"), display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success", log.error
    (sample,) = log.samples
    return sample


def test_time_limit_ends_the_run_and_it_is_scored(tmp_path, monkeypatch):
    async def slow(state):
        state.messages.append(ChatMessageAssistant(content="started"))
        await asyncio.sleep(30)

    _stand_in(monkeypatch, slow)
    sample = _run(tmp_path, monkeypatch, timeout_sec=0.5, reward=1.0)
    assert sample.store[LIMIT_REACHED] == "time"
    score = sample.scores["harbor_reward"]
    assert score.value == 1.0 and score.metadata[LIMIT_REACHED] == "time"
    assert any(m.role == "assistant" and m.text == "started" for m in sample.messages)   # the work so far is kept


def test_turn_limit_ends_the_run_after_the_allowed_turns(tmp_path, monkeypatch):
    _stand_in(monkeypatch, _generate_forever)
    sample = _run(tmp_path, monkeypatch, max_steps=3, scaffold="react")
    assert sample.store[LIMIT_REACHED] == "turns"
    assert sample.store[TURNS] == 3
    assert sample.scores["harbor_reward"].metadata[TURNS] == sample.store[TURNS]
    assert sum(m.role == "assistant" for m in sample.messages) == 3


def test_a_run_that_finishes_reports_its_turns_and_no_limit(tmp_path, monkeypatch):
    async def two_turns(state):
        model = get_model("mockllm/model")
        for _ in range(2):
            state.output = await model.generate(state.messages)
            state.messages.append(state.output.message)

    _stand_in(monkeypatch, two_turns)
    sample = _run(tmp_path, monkeypatch, timeout_sec=60, max_steps=10)
    assert sample.store[LIMIT_REACHED] is None and sample.store[TURNS] == 2


def test_a_native_step_limit_that_is_used_up_counts_as_the_turn_limit(tmp_path, monkeypatch):
    """mini-swe-agent stops cleanly at its own step limit; using every step is a limit run."""
    async def three_turns(state):
        model = get_model("mockllm/model")
        for _ in range(3):
            state.output = await model.generate(state.messages)
            state.messages.append(state.output.message)

    _stand_in(monkeypatch, three_turns)
    assert _run(tmp_path, monkeypatch, max_steps=3, scaffold="mini_swe_agent").store[LIMIT_REACHED] == "turns"
    assert _run(tmp_path, monkeypatch, max_steps=5, scaffold="mini_swe_agent").store[LIMIT_REACHED] is None


def test_no_limits_when_the_task_sets_none(tmp_path, monkeypatch):
    async def one_turn(state):
        state.output = await get_model("mockllm/model").generate(state.messages)

    _stand_in(monkeypatch, one_turn)
    sample = _run(tmp_path, monkeypatch, timeout_sec=None, max_steps=0)
    assert sample.store[LIMIT_REACHED] is None and sample.store[TURNS] == 1


def test_an_agent_that_fails_is_a_sample_error_not_a_score(tmp_path, monkeypatch):
    async def broken(state):
        raise RuntimeError("sandbox went away")

    _stand_in(monkeypatch, broken)

    async def inner(state, target):
        return Score(value=0.0)

    monkeypatch.setattr(reward_module, "_inner_scorer", lambda timeout: inner)
    task = Task(dataset=[Sample(id="s", input="go")], solver=agent_scaffold.agent_solver("react"),
                scorer=reward_module.harbor_reward())
    log = inspect_eval(task, model=get_model("mockllm/model"), display="none", log_dir=str(tmp_path), fail_on_error=False)[0]
    (sample,) = log.samples
    assert sample.error is not None and "sandbox went away" in sample.error.message and not sample.scores


# --- the data on the hub -----------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.parametrize("package, digest, count", [
    ("terminal-bench/terminal-bench-2-1", terminal_bench_module.DATASET_DIGEST, 89),
    ("terminal-bench/terminal-bench", terminal_bench_module.DATASET_DIGEST_4, 66),
    ("scale-ai/swe-bench-pro", swe_bench_module.VARIANTS["pro"][1], 731),
    ("cais/swebenchpro", swe_bench_module.VARIANTS["pro_hardened"][1], 731),
    ("swe-bench/swe-bench-verified", swe_bench_module.VARIANTS["verified"][1], 500),
    ("datacurve/deep-swe-1-1", deep_swe_module.DATASET_DIGEST, 113),
])
def test_pinned_digests_resolve_to_the_expected_task_lists(package, digest, count):
    """Resolving the dataset by digest lists its tasks (no task archive is downloaded)."""
    from inspect_harbor._harbor.hub import HubClient, resolve_hub_dataset

    async def resolve():
        client = HubClient()
        try:
            return await resolve_hub_dataset(f"{package}@{digest}", client=client)
        finally:
            await client.aclose()

    metadata = asyncio.run(resolve())
    assert metadata.version == digest and len(metadata.task_refs) == count and not metadata.yanked_at
