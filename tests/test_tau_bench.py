"""tau-bench: tool conversion, the conversation loop, the reward, pass^k and the data set-up.

The loop and the scorer run against a small domain of this file's own, registered with tau2
(its tools, tasks and environment), so no benchmark data is downloaded; models are ``mockllm``
with scripted outputs. Tests that need tau2 skip when it is not installed; the one that runs a
real airline task in a fresh process is marked ``network`` (it downloads the pinned data once).
"""
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import tempfile

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score, SampleScore
from inspect_ai.solver import Generate, TaskState, solver
from pydantic import BaseModel

from dna_eval import tau2_env
from dna_eval.metrics.pass_hat_k import pass_hat_k
from dna_eval.scorers.tau_reward import tau_reward
from dna_eval.solvers.tau_conversation import _inline, tau_conversation, tool_params

MODEL = "mockllm/model"
GREETING = "Hi! How can I help you today?"
WRITE = {"note_id": "n1", "text": "hi", "address": {"city": "Oslo", "zip": "0150"}}


# --- schema conversion (no tau2 needed) -------------------------------------------------------------


def test_refs_are_inlined_and_unused_keys_dropped():
    schema = {
        "$defs": {"Address": {"title": "Address", "type": "object", "description": "Where.",
                              "properties": {"city": {"type": "string", "description": "City.", "title": "City"}},
                              "required": ["city"]}},
        "type": "object", "title": "parameters",
        "properties": {"title": {"type": "string", "description": "A property that is called title."},
                       "address": {"$ref": "#/$defs/Address", "description": "Postal address."},
                       "stops": {"type": "array", "items": {"$ref": "#/$defs/Address"}, "description": "Stops."}},
        "required": ["title", "address"],
    }
    params = tool_params(schema).model_dump(exclude_none=True)
    assert "$defs" not in str(params) and "$ref" not in str(params) and "'title': 'City'" not in str(params)
    assert params["required"] == ["title", "address"]
    assert set(params["properties"]) == {"title", "address", "stops"}
    assert params["properties"]["address"] == {
        "type": "object", "description": "Postal address.",     # the use site's description wins
        "properties": {"city": {"type": "string", "description": "City."}}, "required": ["city"]}
    assert params["properties"]["stops"]["items"]["properties"]["city"]["type"] == "string"
    assert "additionalProperties" not in params          # tau2's schemas do not forbid extra keys


def test_a_keyword_inspect_cannot_carry_is_an_error():
    with pytest.raises(ValueError, match="oneOf"):
        _inline({"oneOf": [{"type": "string"}]}, {})


# --- pass^k ---------------------------------------------------------------------------------------


def _scores(trials: dict[str, list[float]]) -> list[SampleScore]:
    return [SampleScore(score=Score(value=v), sample_id=sid) for sid, vs in trials.items() for v in vs]


def test_pass_hat_k_is_c_choose_k_over_n_choose_k():
    out = pass_hat_k()(_scores({"a": [1, 1, 1, 1], "b": [1, 0, 1, 0], "c": [0, 0, 0, 0]}))
    assert out == pytest.approx({"pass_hat_1": (1 + 0.5 + 0) / 3, "pass_hat_2": (1 + 1 / 6 + 0) / 3,
                                 "pass_hat_3": 1 / 3, "pass_hat_4": 1 / 3})


def test_pass_hat_k_uses_the_smallest_trial_count():
    out = pass_hat_k()(_scores({"a": [1, 1, 1], "b": [1, 0]}))
    assert out == pytest.approx({"pass_hat_1": 0.75, "pass_hat_2": 0.5})
    assert pass_hat_k()([]) == {}


# --- data set-up (no tau2 needed) -----------------------------------------------------------------


def test_data_is_extracted_from_the_pinned_archive_once(tmp_path, monkeypatch):
    prefix = f"tau2-bench-{tau2_env.TAU2_COMMIT}"
    members = {"data/tau2/domains/airline/db.json": b"{}", "data/tau2/user_simulator/g.md": b"guidelines",
               "data/tau2/results/huge.json": b"x", "data/voice/a.wav": b"x", "README.md": b"x"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(f"{prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()
    downloads = []

    def fake_fetch(url, path, sha256):
        downloads.append((url, sha256))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive)
        return path

    monkeypatch.setenv("DNA_EVAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TAU2_DATA_DIR", "-")
    monkeypatch.delenv("TAU2_DATA_DIR")          # so that the variable ensure_data sets is removed afterwards
    monkeypatch.setattr(tau2_env, "fetch_pinned", fake_fetch)
    data = tau2_env.ensure_data()
    assert data == tmp_path / "tau2" / tau2_env.TAU2_COMMIT / "data" and os.environ["TAU2_DATA_DIR"] == str(data)
    assert sorted(p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()) == [
        "tau2/domains/airline/db.json", "tau2/user_simulator/g.md"]
    assert downloads == [(tau2_env.TAU2_ARCHIVE_URL, tau2_env.TAU2_ARCHIVE_SHA256)]
    monkeypatch.delenv("TAU2_DATA_DIR")
    assert tau2_env.ensure_data() == data and len(downloads) == 1       # second call: no download
    assert not list((tmp_path / "tau2").glob("extract-*"))                # temporary directory removed


def test_a_data_directory_set_by_the_user_is_used_as_it_is(tmp_path, monkeypatch):
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(tau2_env, "fetch_pinned", lambda *a: pytest.fail("must not download"))
    assert tau2_env.ensure_data() == tmp_path


def test_only_offline_retrieval_variants_are_accepted():
    with pytest.raises(ValueError, match="embedding"):
        tau2_env.check_retrieval_variant("banking_knowledge", "alltools")
    with pytest.raises(ValueError):
        tau2_env.check_retrieval_variant("banking_knowledge", "no_such_variant")
    tau2_env.check_retrieval_variant("banking_knowledge", "full_kb")
    tau2_env.check_retrieval_variant("airline", "alltools")              # ignored outside banking_knowledge


# --- a small domain registered with tau2 ----------------------------------------------------------------


class _Address(BaseModel):
    city: str
    zip: str


@pytest.fixture(scope="module")
def toy():
    """tau2 with a toy domain, ``toy``: agent tools ``read_note`` and ``write_note`` (with a nested
    parameter type), the user tool ``look_at_screen``, and three tasks."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TAU2_DATA_DIR", os.environ.get("TAU2_DATA_DIR") or tempfile.mkdtemp(prefix="tau2-data-"))
        pytest.importorskip("tau2")
        from pydantic import Field
        from tau2.data_model.tasks import Action, EvaluationCriteria, RewardType, Task as Tau2Task, UserScenario
        from tau2.environment.db import DB
        from tau2.environment.environment import Environment
        from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
        from tau2.registry import registry

        class ToyDB(DB):
            notes: dict[str, str] = Field(default_factory=dict)

        class ToyUserDB(DB):
            screen: str = "all good"

        class ToyTools(ToolKitBase):
            db: ToyDB

            @is_tool(ToolType.READ)
            def read_note(self, note_id: str) -> str:
                """Read a note.

                Args:
                    note_id: The id of the note.
                """
                return self.db.notes[note_id]

            @is_tool(ToolType.WRITE)
            def write_note(self, note_id: str, text: str, address: _Address) -> str:
                """Write a note.

                Args:
                    note_id: The id of the note.
                    text: What the note says.
                    address: Where it was written.
                """
                self.db.notes[note_id] = f"{text} @ {_Address.model_validate(address).city}"
                return "ok"

        class ToyUserTools(ToolKitBase):
            db: ToyUserDB

            @is_tool(ToolType.READ)
            def look_at_screen(self) -> str:
                """Describe what the screen shows."""
                return self.db.screen

        def get_environment(solo_mode: bool = False) -> Environment:
            return Environment(domain_name="toy", policy="Be kind. Toy policy.", tools=ToyTools(ToyDB()),
                               user_tools=ToyUserTools(ToyUserDB()))

        def get_tasks(task_split_name=None) -> list:
            write = EvaluationCriteria(actions=[Action(action_id="a1", name="write_note", arguments=WRITE)],
                                       reward_basis=[RewardType.DB])
            nl = EvaluationCriteria(nl_assertions=["The agent greets the user."], reward_basis=[RewardType.NL_ASSERTION])
            return [Tau2Task(id=i, user_scenario=UserScenario(instructions=f"Scenario {i}."), evaluation_criteria=c)
                    for i, c in (("write", write), ("nl", nl))]

        registry.register_domain(get_environment, "toy")
        registry.register_tasks(get_tasks, "toy")
        mp.setattr("tau2.user.user_simulator.get_global_user_sim_guidelines",
                   lambda use_tools=False: "GUIDELINES" + (" WITH TOOLS" if use_tools else "") + " <PERSONA_GUIDELINES>")
        yield sys.modules["tau2"]


def say(text: str) -> ModelOutput:
    return ModelOutput.from_content(MODEL, text)


def call(name: str, args: dict, id: str) -> ModelOutput:
    return ModelOutput.for_tool_call(MODEL, name, args, tool_call_id=id)


def run(tmp_path, agent, user, *, task_id="write", grader=None, **solver_args):
    """Run one toy task once with scripted agent, user (and grader) outputs; the finished sample."""
    roles = {"user": get_model(MODEL, custom_outputs=user)}
    if grader is not None:
        roles["grader"] = get_model(MODEL, custom_outputs=grader)
    task = Task(dataset=[Sample(id=task_id, input="", metadata={"domain": "toy", "split": "base", "task_id": task_id})],
                solver=tau_conversation(**solver_args), scorer=tau_reward(), epochs=1)
    log = eval(task, model=get_model(MODEL, custom_outputs=agent), model_roles=roles, log_dir=str(tmp_path),
               display="none")[0]
    assert log.status == "success", log.error
    return log.samples[0]


def roles_of(sample) -> list[str]:
    """Who spoke each message of the tau2 trajectory, tool results as ``tool``."""
    return [m["role"] + ("*" if m.get("tool_calls") else "") for m in sample.store["tau_simulation"]["messages"]]


# --- the conversation ---------------------------------------------------------------------------


def test_scripted_conversation_reaches_the_user_stop(toy, tmp_path):
    s = run(tmp_path,
            agent=[call("write_note", WRITE, "w1"), say("Done, note n1 is written.")],
            user=[say("Please write note n1."), say("Thanks. ###STOP###")])
    assert roles_of(s) == ["assistant", "user", "assistant*", "tool", "assistant", "user"]
    assert s.scores["tau_reward"].value == 1.0
    md = s.scores["tau_reward"].metadata
    assert (md["termination_reason"], md["limit_reached"], md["turns"]) == ("user_stop", None, 2)
    assert md["reward_info"]["db_check"]["db_match"] is True
    # the model under evaluation saw the policy, the greeting, the user's words and the tool result
    assert [m.role for m in s.messages] == ["system", "assistant", "user", "assistant", "tool", "assistant", "user"]
    assert "Toy policy." in s.messages[0].text and s.messages[1].text == GREETING
    assert s.messages[4].text == "ok" and s.messages[4].tool_call_id == "w1"
    # the store holds what the scorer and agent_diagnostics read
    assert s.store["tau_termination"] == "user_stop" and s.store["limit_reached"] is None and s.store["turns"] == 2
    assert s.store["tau_simulation"]["termination_reason"] == "user_stop"


def test_a_conversation_that_does_not_change_the_database_scores_zero(toy, tmp_path):
    s = run(tmp_path, agent=[say("I cannot do that.")], user=[say("Write note n1."), say("###STOP###")])
    assert s.scores["tau_reward"].value == 0.0
    assert s.scores["tau_reward"].metadata["reward_info"]["db_check"]["db_match"] is False


def test_the_user_model_is_prompted_as_tau2s_simulator_does(toy, tmp_path):
    s = run(tmp_path, agent=[say("Hello again.")], user=[say("Hi."), say("###STOP###")])
    user_calls = [e for e in s.events if e.event == "model" and e.role == "user"]
    assert len(user_calls) == 2
    first = [(m.role, m.text) for m in user_calls[0].input]
    assert first[0][0] == "system" and first[0][1].startswith("GUIDELINES WITH TOOLS") and "Scenario write." in first[0][1]
    assert first[1:] == [("user", GREETING)]                      # roles flipped: the greeting is the "user" turn
    assert [(m.role, m.text) for m in user_calls[1].input][-2:] == [("assistant", "Hi."), ("user", "Hello again.")]
    assert [t.name for t in user_calls[0].tools] == ["look_at_screen"]


def test_the_user_may_call_its_own_tools(toy, tmp_path):
    s = run(tmp_path, agent=[say("Is the screen fine?")],
            user=[call("look_at_screen", {}, "u1"), say("It says all good."), say("###STOP###")])
    sim = s.store["tau_simulation"]["messages"]
    assert roles_of(s) == ["assistant", "user*", "tool", "user", "assistant", "user"]
    assert sim[1]["tool_calls"][0]["requestor"] == "user" and sim[2]["requestor"] == "user"
    assert sim[2]["content"] == "all good"
    assert [m.role for m in s.messages] == ["system", "assistant", "user", "assistant", "user"]   # the agent never sees them
    assert s.messages[2].text == "It says all good."


def test_the_agent_sees_the_official_tool_schemas(toy, tmp_path):
    s = run(tmp_path, agent=[say("Hello.")], user=[say("Hi."), say("###STOP###")])
    agent_call = next(e for e in s.events if e.event == "model" and e.role is None)
    tools = {t.name: t for t in agent_call.tools}
    assert set(tools) == {"read_note", "write_note"}
    props = tools["write_note"].parameters.model_dump(exclude_none=True)
    assert props["required"] == ["note_id", "text", "address"]
    assert props["properties"]["address"]["properties"]["city"]["type"] == "string"
    assert "Where it was written." in props["properties"]["address"]["description"]


def test_step_limit_ends_the_conversation(toy, tmp_path):
    s = run(tmp_path, agent=[say("ok")] * 5, user=[say("hm")] * 5, max_steps=3)
    assert roles_of(s) == ["assistant", "user", "assistant", "user"]          # three steps after the greeting
    md = s.scores["tau_reward"].metadata
    assert (md["termination_reason"], md["limit_reached"]) == ("max_steps", "steps")
    assert s.scores["tau_reward"].value == 0.0


def test_a_user_stop_at_the_step_limit_counts_as_the_limit(toy, tmp_path):
    """As in tau2's orchestrator, the limit check runs after the stop and overrides its reason."""
    s = run(tmp_path, agent=[say("ok")], user=[say("###STOP###")], max_steps=1)
    assert s.store["tau_termination"] == "max_steps" and s.store["limit_reached"] == "steps"


def test_failed_tool_calls_end_the_conversation_at_the_error_limit(toy, tmp_path):
    s = run(tmp_path, agent=[call("no_such_tool", {}, f"c{i}") for i in range(5)], user=[say("go")] * 3, max_errors=2)
    sim = s.store["tau_simulation"]["messages"]
    assert [m["role"] for m in sim].count("tool") == 2 and all(m["error"] for m in sim if m["role"] == "tool")
    assert sim[-1]["content"].startswith("Error:")
    assert (s.store["tau_termination"], s.store["limit_reached"]) == ("too_many_errors", "errors")
    assert [m.role for m in s.messages][-1] == "tool"                    # the model saw each error text


def test_an_empty_agent_message_is_an_agent_error_not_a_crash(toy, tmp_path):
    s = run(tmp_path, agent=[say("")], user=[say("Hi.")])
    assert (s.store["tau_termination"], s.store["limit_reached"]) == ("agent_error", None)
    assert s.scores["tau_reward"].value == 0.0 and s.scores["tau_reward"].metadata["turns"] == 1


def test_a_cut_off_context_is_a_limit(toy, tmp_path):
    cut = ModelOutput.from_content(MODEL, "partial", stop_reason="model_length")
    s = run(tmp_path, agent=[cut], user=[say("Hi.")])
    assert (s.store["tau_termination"], s.store["limit_reached"]) == ("context_window_exceeded", "context")
    assert s.scores["tau_reward"].value == 0.0


def failed_sample(tmp_path, agent, user=None):
    """Run the toy task expecting the sample to error; the sample."""
    task = Task(dataset=[Sample(id="write", input="", metadata={"domain": "toy", "split": "base", "task_id": "write"})],
                solver=tau_conversation(), scorer=tau_reward(), epochs=1)
    roles = {} if user is None else {"user": get_model(MODEL, custom_outputs=user)}
    log = eval(task, model=get_model(MODEL, custom_outputs=agent), model_roles=roles, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "error"
    return log.samples[0]


def test_an_empty_user_message_is_a_sample_error(toy, tmp_path):
    s = failed_sample(tmp_path, agent=[say("x")], user=[say("")])
    assert "empty message" in s.error.message and not s.scores


def test_missing_user_role_stops_the_run(toy, tmp_path):
    assert "role 'user' is required" in failed_sample(tmp_path, agent=[say("x")]).error.message


# --- the scorer on recorded conversations -----------------------------------------------------------


def recorded(toy, messages, reason="user_stop") -> dict:
    """A tau2 SimulationRun, as JSON, for these messages."""
    from tau2.data_model.simulation import SimulationRun
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
    built = []
    for kind, *rest in messages:
        if kind == "a":
            built.append(AssistantMessage(role="assistant", content=rest[0]))
        elif kind == "u":
            built.append(UserMessage(role="user", content=rest[0]))
        elif kind == "call":
            built.append(AssistantMessage(role="assistant", tool_calls=[ToolCall(id=rest[0], name=rest[1], arguments=rest[2])]))
        else:
            built.append(ToolMessage(id=rest[0], role="tool", content=rest[1]))
    sim = SimulationRun(id="r", task_id="write", start_time="t", end_time="t", duration=1.0,
                        termination_reason=reason, messages=built)
    return sim.model_dump(mode="json")


@solver
def replay(simulation: dict, task_id: str):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set("tau_simulation", simulation)
        state.store.set("tau_termination", simulation["termination_reason"])
        return state
    return solve


def score_recorded(tmp_path, simulation, task_id="write", grader=None):
    roles = {"grader": get_model(MODEL, custom_outputs=grader)} if grader is not None else {}
    task = Task(dataset=[Sample(id=task_id, input="", metadata={"domain": "toy", "split": "base", "task_id": task_id})],
                solver=replay(simulation, task_id), scorer=tau_reward(), epochs=1)
    log = eval(task, model=get_model(MODEL), model_roles=roles, log_dir=str(tmp_path), display="none")[0]
    return read_eval_log(log.location, resolve_attachments=True).samples[0]     # long prompts are stored as attachments


def test_a_recorded_trajectory_is_replayed_and_graded(toy, tmp_path):
    good = [("a", GREETING), ("u", "write it"), ("call", "c1", "write_note", WRITE), ("tool", "c1", "ok"), ("a", "done"), ("u", "###STOP###")]
    assert score_recorded(tmp_path / "good", recorded(toy, good)).scores["tau_reward"].value == 1.0
    wrong = [("a", GREETING), ("u", "write it"), ("call", "c1", "write_note", {**WRITE, "text": "bye"}),
             ("tool", "c1", "ok"), ("a", "done"), ("u", "###STOP###")]
    assert score_recorded(tmp_path / "wrong", recorded(toy, wrong)).scores["tau_reward"].value == 0.0


def test_only_a_user_stop_can_score(toy, tmp_path):
    good = [("a", GREETING), ("u", "write it"), ("call", "c1", "write_note", WRITE), ("tool", "c1", "ok")]
    s = score_recorded(tmp_path, recorded(toy, good, reason="max_steps"))
    assert s.scores["tau_reward"].value == 0.0
    assert s.scores["tau_reward"].metadata["reward_info"]["info"]["note"].startswith("Simulation terminated prematurely")


def test_a_replay_that_disagrees_with_the_recorded_tool_output_is_an_error(toy, tmp_path):
    """tau2 raises; the sample errors (and is retried with --retry-on-error) instead of scoring 0."""
    bad = [("a", GREETING), ("u", "go"), ("call", "c1", "write_note", WRITE), ("tool", "c1", "a different answer"), ("u", "###STOP###")]
    task = Task(dataset=[Sample(id="write", input="", metadata={"domain": "toy", "split": "base", "task_id": "write"})],
                solver=replay(recorded(toy, bad), "write"), scorer=tau_reward(), epochs=1)
    log = eval(task, model=get_model(MODEL), log_dir=str(tmp_path), display="none")[0]
    assert log.status == "error" and "a different answer" in log.samples[0].error.message


NL_CONVERSATION = [("a", GREETING), ("u", "hello"), ("a", "Welcome!"), ("u", "###STOP###")]
VERDICT = '{"results": [{"expectedOutcome": "The agent greets the user.", "reasoning": "It said welcome.", "metExpectation": %s}]}'


@pytest.mark.parametrize("reply, reward", [
    (VERDICT % "true", 1.0), (VERDICT % "false", 0.0),
    ("Here you go:\n```json\n" + VERDICT % "true" + "\n```", 1.0)])
def test_natural_language_assertions_are_judged_by_the_grader_role(toy, tmp_path, reply, reward):
    s = score_recorded(tmp_path, recorded(toy, NL_CONVERSATION), task_id="nl", grader=[say(reply)])
    score = s.scores["tau_reward"]
    assert score.value == reward
    checks = score.metadata["reward_info"]["nl_assertions"]
    assert [c["nl_assertion"] for c in checks] == ["The agent greets the user."]
    judge_calls = [e for e in s.events if e.event == "model" and e.role == "grader"]
    assert len(judge_calls) == 1
    text = judge_calls[0].input[-1].text
    assert "assistant: Welcome!" in text and "The agent greets the user." in text    # tau2's own prompt, unchanged
    assert judge_calls[0].config.temperature == 0.0


def test_a_judge_reply_without_json_is_asked_for_again_then_errors(toy, tmp_path):
    ok = VERDICT % "true"
    s = score_recorded(tmp_path / "retry", recorded(toy, NL_CONVERSATION), task_id="nl",
                       grader=[say("I think so."), say(ok)])
    assert s.scores["tau_reward"].value == 1.0
    task = Task(dataset=[Sample(id="nl", input="", metadata={"domain": "toy", "split": "base", "task_id": "nl"})],
                solver=replay(recorded(toy, NL_CONVERSATION), "nl"), scorer=tau_reward(), epochs=1)
    log = eval(task, model=get_model(MODEL), model_roles={"grader": get_model(MODEL, custom_outputs=[say("no")] * 3)},
               log_dir=str(tmp_path / "fail"), display="none")[0]
    assert log.status == "error" and "did not return JSON" in log.samples[0].error.message


def test_tasks_without_assertions_need_no_grader(toy, tmp_path):
    good = [("a", GREETING), ("u", "write it"), ("call", "c1", "write_note", WRITE), ("tool", "c1", "ok"), ("u", "###STOP###")]
    assert score_recorded(tmp_path, recorded(toy, good)).scores["tau_reward"].value == 1.0     # no grader bound


def test_assertion_tasks_require_the_grader_role(toy, tmp_path):
    task = Task(dataset=[Sample(id="nl", input="", metadata={"domain": "toy", "split": "base", "task_id": "nl"})],
                solver=replay(recorded(toy, NL_CONVERSATION), "nl"), scorer=tau_reward(), epochs=1)
    log = eval(task, model=get_model(MODEL), log_dir=str(tmp_path), display="none")[0]
    assert log.status == "error" and "role 'grader' is required" in log.samples[0].error.message


# --- real data -------------------------------------------------------------------------------------


@pytest.mark.network
def test_real_airline_task_runs_in_a_fresh_process(tmp_path):
    """Downloads the pinned archive once (about 90 MB), then runs airline task 0 with scripted models."""
    if importlib.util.find_spec("tau2") is None:       # find_spec does not import it
        pytest.skip("tau2 is not installed")
    code = """
import json, sys
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model
from dna_eval.tasks.agentic.tau_bench import tau_bench
M = "mockllm/model"
agent = get_model(M, custom_outputs=[
    ModelOutput.for_tool_call(M, "get_user_details", {"user_id": "emma_kim_9957"}, tool_call_id="c1"),
    ModelOutput.from_content(M, "I am sorry, that reservation cannot be cancelled.")])
user = get_model(M, custom_outputs=[ModelOutput.from_content(M, "Cancel EHGLP3, I am emma_kim_9957."),
                                    ModelOutput.from_content(M, "###STOP###")])
task = tau_bench(epochs=1)
assert len(task.dataset) == 50
log = eval(task, model=agent, model_roles={"user": user}, sample_id="0", log_dir=sys.argv[1], display="none")[0]
s = log.samples[0]
print(json.dumps({"status": log.status, "reward": s.scores["tau_reward"].value, "tools": len(s.store["tau_simulation"]["messages"])}))
"""
    env = {k: v for k, v in os.environ.items() if k != "TAU2_DATA_DIR"}
    try:
        done = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        pytest.skip("download too slow")
    assert done.returncode == 0, done.stderr[-2000:]
    assert '"status": "success"' in done.stdout and '"reward": 1.0' in done.stdout
