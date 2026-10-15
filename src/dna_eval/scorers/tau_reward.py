"""``tau_reward``: the tau-bench reward of a finished conversation, from tau2-bench's own evaluator.

The solver (``tau_conversation``) leaves the conversation as tau2's ``SimulationRun`` in
``state.store``. This scorer rebuilds it and calls ``tau2.evaluator.evaluator.evaluate_simulation``
with ``EvaluationType.ALL`` (github.com/sierra-research/tau2-bench, v1.0.1, MIT): the evaluator
replays the agent's tool calls on a fresh environment and compares the resulting database with
the one the task's reference actions produce, checks the required actions, the information the
agent had to communicate and the environment assertions, as the task's ``reward_basis`` says, and
multiplies the parts. The score is that reward (1.0 or 0.0); a conversation that did not end with
a user stop scores 0.

Natural-language assertions (40 retail tasks and 1 ``banking_knowledge`` task) are judged by an
LLM. tau2 hard-codes that call (``gpt-4.1`` through litellm); here its prompt is left as it is
and the call is answered by the model bound to the ``grader`` role (required, and only looked up
for a task that has such assertions), at temperature 0 as tau2 sets it. A reply that is not JSON
(tau2 parses it unprepared) is read from inside a code fence or prose, and asked for again up to
three times before the sample errors. A different judge shifts the rewards of those tasks; compare
only runs with the same judge.

``Score.metadata``: ``reward_info`` (tau2's ``RewardInfo`` as JSON: database check, action checks,
communicate checks, assertions, breakdown), ``termination_reason``, and ``limit_reached`` and
``turns`` copied from the store for ``agent_diagnostics``. An evaluator or judge call that fails
raises, so the sample is an error to retry, not a 0.
"""
from __future__ import annotations

import asyncio
import contextvars
import json

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval import tau2_env
from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS, agent_diagnostics
from dna_eval.metrics.pass_hat_k import pass_hat_k
from dna_eval.roles import GRADER_ROLE
from dna_eval.utils.json import escape_stray_backslashes, extract_json

JUDGE_ATTEMPTS = 3

# The grader the running sample's NL-assertion calls go to, with the loop to run them on: the
# evaluator is synchronous and runs in a worker thread (it replays the conversation), and the
# worker thread copies the sample's context, so each sample finds its own judge here.
_judge: contextvars.ContextVar = contextvars.ContextVar("tau_nl_judge", default=None)
_patched = False


def _reply_json(reply: str) -> str:
    """``reply`` as a JSON string, read from a code fence or surrounding prose; ``ValueError`` if it has none."""
    try:
        return json.dumps(extract_json(reply))
    except ValueError:
        return json.dumps(extract_json(escape_stray_backslashes(reply)))


def _nl_generate(model, messages, tools=None, tool_choice=None, call_name=None, **kwargs):
    """Stands in for ``tau2.utils.llm_utils.generate`` inside tau2's NL-assertion evaluator:
    sends its messages to the sample's grader and returns an ``AssistantMessage`` whose content is JSON."""
    from tau2.data_model.message import AssistantMessage, SystemMessage

    loop, grader = _judge.get()
    chat = [ChatMessageSystem(content=m.content) if isinstance(m, SystemMessage) else ChatMessageUser(content=m.content)
            for m in messages]
    error = ""
    for _ in range(JUDGE_ATTEMPTS):
        reply = asyncio.run_coroutine_threadsafe(grader.generate(chat), loop).result().completion
        try:
            return AssistantMessage(role="assistant", content=_reply_json(reply))
        except ValueError as e:
            error = f"{e}: {reply[:200]!r}"
    raise RuntimeError(f"the grader did not return JSON for the natural-language assertions: {error}")


def _route_nl_judge() -> None:
    """Make tau2's NL-assertion evaluator use ``_nl_generate`` when a grader is set for the sample."""
    global _patched
    if _patched:
        return
    import tau2.evaluator.evaluator_nl_assertions as nl
    original = nl.generate
    nl.generate = lambda *a, **k: (_nl_generate if _judge.get() else original)(*a, **k)
    _patched = True


@scorer(metrics=[accuracy(), stderr(), agent_diagnostics(), pass_hat_k()])
def tau_reward() -> Scorer:
    """Reward of a ``tau_conversation`` run, as tau2's evaluator computes it (see the module docstring)."""

    async def score(state: TaskState, target: Target) -> Score:
        tau2_env.tau2_ready()
        from tau2.data_model.simulation import SimulationRun
        from tau2.data_model.tasks import RewardType
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        md = state.metadata
        domain = md["domain"]
        task = tau2_env.get_task(domain, md["split"], md["task_id"])
        simulation = SimulationRun.model_validate(state.store.get(tau2_env.SIMULATION_KEY))
        criteria = task.evaluation_criteria
        needs_judge = bool(criteria and RewardType.NL_ASSERTION in criteria.reward_basis and criteria.nl_assertions)
        token = None
        if needs_judge:
            _route_nl_judge()
            grader = get_model(role=GRADER_ROLE, required=True, config=GenerateConfig(temperature=0.0))
            token = _judge.set((asyncio.get_running_loop(), grader))
        try:
            info = await asyncio.to_thread(
                evaluate_simulation, simulation=simulation, task=task, evaluation_type=EvaluationType.ALL,
                solo_mode=False, domain=domain,
                env_kwargs=tau2_env.env_kwargs(domain, md.get("retrieval_variant"), task))
        finally:
            if token is not None:
                _judge.reset(token)
        reason = state.store.get(tau2_env.TERMINATION_KEY)
        return Score(
            value=float(info.reward), explanation=f"reward {info.reward} ({reason})",
            metadata={"reward_info": info.model_dump(mode="json"), "termination_reason": reason,
                      LIMIT_REACHED: state.store.get(LIMIT_REACHED), TURNS: state.store.get(TURNS)})

    return score
