"""``tau_conversation``: one tau-bench conversation between the model under evaluation, a simulated
user and the domain's environment.

The environment, its tools and policy, the agent's system prompt, the user simulator's prompt and
the first message are tau2-bench's own (github.com/sierra-research/tau2-bench, v1.0.1, MIT),
imported rather than copied. What this solver replaces is the loop of tau2's ``Orchestrator``
(``orchestrator/orchestrator.py``) and the two language-model calls, which go through Inspect:

- the **agent** is the model under evaluation, called with the environment's tools as Inspect tool
  definitions (the official JSON schemas, with ``$ref`` indirections inlined and the ``title`` keys
  Inspect's schema type has no field for dropped);
- the **user** is the model bound to the ``user`` role (required), prompted exactly as
  ``UserSimulator`` prompts it (system prompt, roles flipped, tools for dual-control domains) so
  every user turn is a model call in the Inspect log;
- tool calls run in the environment (``Environment.get_response``), and its answers, errors
  included, are what the model sees and what the grader later replays.

The loop follows the orchestrator: the agent opens with "Hi! How can I help you today?"; the
user and the agent then alternate; an agent turn is a text message or tool calls (all of one
message run before the next turn); a message from either side, and a batch of tool results,
counts as one step. The conversation ends when the user sends ``###STOP###``,
``###TRANSFER###`` or ``###OUT-OF-SCOPE###``, after ``max_steps`` steps, or once ``max_errors``
tool calls have failed. The standard agent has no stop token of its own.

Where this differs from the orchestrator:

- an agent turn with neither text nor tool calls ends the conversation as ``agent_error`` (the
  orchestrator raises); a turn the model's context window cuts off ends it as
  ``context_window_exceeded`` (a termination reason tau2 defines but its runner never sets). Both
  score 0, like every ending other than a user stop;
- tool-call arguments Inspect could not parse as JSON are run as empty arguments, which the
  environment answers with its own error (the official client raises and drops the run);
- an empty message from the user model raises, making the sample an error to retry with
  ``--retry-on-error``;
- no per-trial seed is passed to the models.

Writes ``state.store``: ``tau_simulation`` (tau2's ``SimulationRun`` as JSON, the input of tau2's
evaluator), ``tau_termination`` (how the conversation ended), and ``limit_reached`` (``"steps"``,
``"errors"``, ``"context"`` or ``None``) and ``turns`` (agent model calls) for ``agent_diagnostics``.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from copy import deepcopy
from typing import Any

from inspect_ai.model import (ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool,
                              ChatMessageUser, GenerateConfig, get_model)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall as InspectToolCall, ToolDef, ToolInfo, ToolParams
from inspect_ai.util import JSONSchema

from dna_eval import tau2_env
from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS
from dna_eval.roles import USER_ROLE

USER_RETRIES = 3        # retries of a failed request to the user model

# JSON-schema keywords Inspect's schema type keeps; ``title`` and the reference keywords are handled here.
_KEPT = set(JSONSchema.model_fields)
_LIMITS = {"max_steps": "steps", "too_many_errors": "errors", "context_window_exceeded": "context"}


def _inline(node: Any, defs: dict[str, Any]) -> Any:
    """``node`` (a JSON schema) with every ``$ref`` replaced by its definition, ``$defs`` and
    ``title`` removed. Raises ``ValueError`` on a keyword Inspect's schema type cannot carry, so
    a schema is never degraded silently."""
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        merged = {**_inline(defs[node["$ref"].rsplit("/", 1)[-1]], defs), **{k: v for k, v in node.items() if k != "$ref"}}
        return _inline(merged, defs) if "$ref" in merged else merged
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "title"):
            continue
        if key not in _KEPT:
            raise ValueError(f"tool schema keyword {key!r} cannot be passed to Inspect")
        if key == "properties":                    # names, not keywords
            out[key] = {name: _inline(v, defs) for name, v in value.items()}
        elif key in ("default", "enum", "examples"):
            out[key] = value
        else:
            out[key] = _inline(value, defs)
    return out


def tool_params(schema: dict[str, Any]) -> ToolParams:
    """Inspect ``ToolParams`` for a tau2 tool's ``openai_schema["function"]["parameters"]``."""
    clean = _inline(schema, schema.get("$defs", {}))
    return ToolParams(properties={k: JSONSchema.model_validate(v) for k, v in clean.get("properties", {}).items()},
                      required=clean.get("required", []), additionalProperties=None)


def tool_infos(tau2_tools: list) -> list[ToolInfo]:
    """Inspect ``ToolInfo`` carrying the schemas of tau2 ``Tool`` objects, as they are (Inspect's
    ``Tool`` type insists on a description for every parameter, which a few of tau2's lack)."""
    return [ToolInfo(name=fn["name"], description=fn.get("description", ""), parameters=tool_params(fn["parameters"]))
            for fn in (t.openai_schema["function"] for t in tau2_tools)]


def inspect_tools(tau2_tools: list) -> list[Tool]:
    """Inspect tools carrying the schemas of tau2 ``Tool`` objects. They are never executed by
    Inspect (the loop runs calls in the environment), so each one's callable only raises. Every
    parameter of the agent's tools has a description, which ``Tool`` requires."""
    tools = []
    for info in tool_infos(tau2_tools):

        async def execute(**_: Any) -> str:       # a new function per tool: ToolDef registers it
            raise RuntimeError("tau-bench tools run in the environment, not through Inspect")

        tools.append(ToolDef(execute, name=info.name, description=info.description, parameters=info.parameters).as_tool())
    return tools


def _text_or_none(text: str) -> str | None:
    return text if text.strip() else None


@solver
def tau_conversation(max_steps: int = 200, max_errors: int = 10, user_temperature: float | None = 0.0) -> Solver:
    """Run the tau-bench conversation for the sample's task (see the module docstring).

    The sample's metadata names the task: ``domain``, ``split``, ``task_id`` and, for
    ``banking_knowledge``, ``retrieval_variant``. ``max_steps`` and ``max_errors`` are the
    official limits (200 and 10). The user model is called with ``user_temperature``
    (0.0 as in the official setup; ``None`` leaves sampling to the server).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        tau2_env.tau2_ready()
        from tau2.agent.llm_agent import LLMAgent
        from tau2.data_model.message import (AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage)
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
        from tau2.runner import build_environment
        from tau2.user.user_simulator import UserSimulator
        from tau2.utils.utils import get_now

        md = state.metadata
        domain, split, task_id = md["domain"], md["split"], md["task_id"]
        task = tau2_env.get_task(domain, split, task_id)
        initial = task.initial_state
        if initial is not None and initial.message_history:
            raise NotImplementedError("tasks that start from a message history are not supported")

        # --- environment, agent prompt, user simulator -------------------------------------------
        kwargs = tau2_env.env_kwargs(domain, md.get("retrieval_variant"), task)
        env = await asyncio.to_thread(build_environment, domain, solo_mode=False, env_kwargs=kwargs)
        await asyncio.to_thread(
            env.set_state,
            initialization_data=initial.initialization_data if initial is not None else None,
            initialization_actions=initial.initialization_actions if initial is not None else None,
            message_history=[])
        try:
            user_tau2_tools = env.get_user_tools(include=task.user_tools) or None
        except Exception:                           # the domain has no user tools
            user_tau2_tools = None
        agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="inspect")
        simulator = UserSimulator(llm="inspect", instructions=str(task.user_scenario), tools=user_tau2_tools)
        user_state = simulator.get_init_state()
        user_tools = tool_infos(user_tau2_tools) if user_tau2_tools else []
        user_model = get_model(role=USER_ROLE, required=True, config=GenerateConfig(
            max_retries=USER_RETRIES, **({} if user_temperature is None else {"temperature": user_temperature})))

        state.tools = inspect_tools(env.get_tools())
        state.tool_choice = "auto"

        # --- tau2 message -> Inspect message, for the user model's view --------------------------
        call_names: dict[str, str] = {}

        def chat(messages: list) -> list[ChatMessage]:
            out: list[ChatMessage] = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    out.append(ChatMessageSystem(content=m.content or ""))
                elif isinstance(m, UserMessage):
                    out.append(ChatMessageUser(content=m.content or ""))
                elif isinstance(m, AssistantMessage):
                    calls = [InspectToolCall(id=c.id, function=c.name, arguments=c.arguments) for c in m.tool_calls or []]
                    call_names.update({c.id: c.function for c in calls})
                    out.append(ChatMessageAssistant(content=m.content or "", tool_calls=calls or None))
                elif isinstance(m, ToolMessage):
                    out.append(ChatMessageTool(content=m.content or "", tool_call_id=m.id, function=call_names.get(m.id)))
            return out

        # --- the conversation ---------------------------------------------------------------------
        first = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
        first.timestamp = get_now()
        state.messages = [ChatMessageSystem(content=agent.system_prompt), ChatMessageAssistant(content=first.content)]
        trajectory: list = [first]
        message: Any = first                     # the message (or list of tool results) just produced
        from_role, to_role = "agent", "user"
        step_count = num_errors = turns = 0
        reason: TerminationReason | None = None
        started, clock = get_now(), time.perf_counter()

        while True:
            if to_role == "user":                # agent or environment -> user
                if isinstance(message, list):
                    user_state.messages.extend(message)
                else:
                    user_state.messages.append(message)
                output = await user_model.generate(chat(user_state.system_messages + user_state.flip_roles()),
                                                   tools=user_tools)
                calls = output.message.tool_calls
                user_msg = UserMessage(
                    role="user", content=_text_or_none(output.message.text),
                    tool_calls=[ToolCall(id=c.id, name=c.function, arguments=c.arguments or {}, requestor="user")
                                for c in calls] if calls else None)
                if not (user_msg.has_content() or user_msg.is_tool_call()):
                    raise RuntimeError(f"the user model returned an empty message (stop reason: {output.stop_reason})")
                user_state.messages.append(user_msg)
                if UserSimulator.is_stop(user_msg):
                    reason = TerminationReason.USER_STOP
                trajectory.append(user_msg)
                message, from_role = user_msg, "user"
                to_role = "env" if user_msg.is_tool_call() else "agent"
                if to_role == "agent":
                    state.messages.append(ChatMessageUser(content=user_msg.content))
            elif to_role == "agent":             # user or environment -> agent
                await generate(state, tool_calls="none")
                turns += 1
                out = state.output
                if out.stop_reason == "model_length":
                    reason = TerminationReason.CONTEXT_WINDOW_EXCEEDED
                else:
                    calls = out.message.tool_calls
                    agent_msg = AssistantMessage(
                        role="assistant", content=_text_or_none(out.message.text),
                        tool_calls=[ToolCall(id=c.id, name=c.function, arguments=c.arguments or {}, requestor="assistant")
                                    for c in calls] if calls else None)
                    if not (agent_msg.has_content() or agent_msg.is_tool_call()):
                        reason = TerminationReason.AGENT_ERROR
                    else:
                        trajectory.append(agent_msg)
                        message, from_role = agent_msg, "agent"
                        to_role = "env" if agent_msg.is_tool_call() else "user"
            else:                                # agent or user -> environment
                results = []
                for call in message.tool_calls:
                    result = await asyncio.to_thread(env.get_response, call)
                    num_errors += bool(result.error)
                    results.append(result)
                    if from_role == "agent":
                        state.messages.append(ChatMessageTool(content=result.content or "", tool_call_id=call.id,
                                                              function=call.name))
                trajectory.extend(results)
                message, to_role, from_role = results, from_role, "env"
            if reason in (TerminationReason.AGENT_ERROR, TerminationReason.CONTEXT_WINDOW_EXCEEDED):
                break
            step_count += 1
            if to_role != "env":                 # as the orchestrator: limits are checked between exchanges
                if step_count >= max_steps:
                    reason = TerminationReason.MAX_STEPS
                if num_errors >= max_errors:
                    reason = TerminationReason.TOO_MANY_ERRORS
            if reason is not None:
                break

        for i, m in enumerate(trajectory):
            m.turn_idx = i
        simulation = SimulationRun(
            id=str(uuid.uuid4()), task_id=task.id, start_time=started, end_time=get_now(),
            duration=time.perf_counter() - clock, termination_reason=reason.value, messages=trajectory)
        state.store.set(tau2_env.SIMULATION_KEY, simulation.model_dump(mode="json"))
        state.store.set(tau2_env.TERMINATION_KEY, reason.value)
        state.store.set(LIMIT_REACHED, _LIMITS.get(reason.value))
        state.store.set(TURNS, turns)
        return state

    return solve
