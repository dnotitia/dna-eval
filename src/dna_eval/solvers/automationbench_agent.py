"""``automationbench_agent``: the AutomationBench agent loop, run with Inspect's tool calling.

AutomationBench (github.com/zapier/AutomationBench, MIT) gives the agent an in-process
simulated business workspace (``WorldState``: CRM, mail, calendar, spreadsheets, chat, ...) and
tools that read and change it. This solver builds the sample's ``WorldState`` and tools with the
benchmark's own code and drives the loop with the model under evaluation, as the official
runner (``automationbench.runner.AutomationBenchEnv``, a ``verifiers`` ``StatefulToolEnv``)
does:

- the three toolsets are the official ones: ``api`` (``api_search``, ``api_fetch``,
  ``base64_encode``; the official CLI default), ``zapier`` (``search_tools`` and
  ``execute_tool``) and ``limited_zapier`` (only the Zapier tools the task lists);
- the ``world`` argument is closed over and hidden from the model; a tool argument that is an
  empty object is dropped, as the official runner does; an exception raised by a tool is
  returned to the model as the tool's output, never raised;
- the loop ends when a model reply has no tool call, or after ``max_turns`` replies (the
  official ``--max-steps``, 50). As in the official loop, the tool calls of the last permitted
  reply are not run.

Deviations from the official runner, all in how the tools are exposed to the model:

- The loop is Inspect's: the model is called through Inspect, so the task's generation config
  applies and the official runner's provider-specific request options (reasoning effort,
  prompt caching, batch transport) do not.
- Inspect checks each tool call against the tool's parameter schema before running it, and
  returns a validation error to the model on a mismatch; the official runner calls the Python
  function directly. The schemas are the official ones (generated from the functions as the
  official runner does) except that ``required`` lists only the parameters without a Python
  default (the official schema lists every parameter, for strict decoding, while the function
  accepts omissions), so an omitted optional argument runs as it does officially. An empty
  object given for an optional parameter is rejected by the schema rather than dropped.
  Inspect also requires every parameter to carry a description; those the official schema
  leaves without one (about half of the Zapier tools' parameters) are given their schema title
  (``"Max Results"``).
- Tool outputs are not truncated (Inspect would cut them at 16 KiB by default; the official
  runner does not).
- A model reply that ends because the context window is full ends the run, recorded as
  ``limit_reached = "context"``; the official runner ends it as an overlong prompt.

``state.store`` keys written (read by ``scorers/automationbench_state.py``, which does not
import this module):

- ``automationbench_world``: the final ``WorldState`` as ``model_dump(mode="json")``;
- ``limit_reached``: ``"turns"`` when the turn cap ended the run, ``"context"`` when the context
  window did, else ``None``; ``turns``: the number of model replies.

The sample's ``metadata["info"]`` holds the task's ``initial_state``, ``assertions`` and
``zapier_tools`` (``tasks/agentic/automationbench.py`` writes it).
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from inspect_ai.model import execute_tools
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolParams

from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS

TOOLSETS = ("api", "zapier", "limited_zapier")
MAX_TURNS = 50                       # the official --max-steps default

WORLD_KEY = "automationbench_world"  # ``state.store`` key holding the final world (JSON)
INFO_KEY = "info"                    # ``state.metadata`` key holding the task's info dict

NO_TRUNCATION = 1 << 30              # bytes; tool output is passed to the model whole


def _skips_world(func: Callable) -> bool:
    return "world" in inspect.signature(func).parameters


@functools.lru_cache(maxsize=None)
def _official_schema(func: Callable) -> tuple[str, str, dict[str, Any]]:
    """``(name, description, JSON schema)`` of a tool function, produced as the official runner
    produces them: the function with ``world`` removed from its signature, converted by
    ``verifiers``."""
    from verifiers.envs.stateful_tool_env import filter_signature
    from verifiers.utils.tool_utils import convert_func_to_tool_def

    visible = filter_signature(func, ["world"] if _skips_world(func) else [])
    tool_def = convert_func_to_tool_def(visible)
    return tool_def.name, tool_def.description, tool_def.parameters


def _tool_def(func: Callable, world: Any) -> ToolDef:
    """An Inspect tool for ``func`` that injects ``world`` and answers errors with their text."""
    name, description, schema = _official_schema(func)
    params = ToolParams.model_validate(schema)
    for p, prop in params.properties.items():             # Inspect requires a description on every parameter
        prop.description = prop.description or schema["properties"][p].get("title", p)
    params.required = [p for p, v in inspect.signature(func).parameters.items()
                       if p in params.properties and v.default is inspect.Parameter.empty]
    inject_world = _skips_world(func)

    async def execute(**kwargs: Any) -> str:
        args = {k: v for k, v in kwargs.items() if not (isinstance(v, dict) and not v)}
        if inject_world:
            args["world"] = world
        try:
            result = func(**args)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, str) else str(result)
        except Exception as e:  # noqa: BLE001  (the official runner returns the text of any error)
            return f"{e}"

    return ToolDef(execute, name=name, description=description, parameters=params)


def _tool_functions(toolset: str, info: dict) -> list[Callable]:
    """The functions the official runner registers for ``toolset`` (and, for
    ``limited_zapier``, for this task)."""
    if toolset == "api":
        from automationbench.tools.api import API_TOOLS
        return list(API_TOOLS)
    if toolset == "zapier":
        from automationbench.tools.zapier.meta import execute_tool, search_tools
        return [search_tools, execute_tool]
    from automationbench.tools import ALL_TOOLS
    wanted = info.get("zapier_tools", [])
    unknown = set(wanted) - {t.__name__ for t in ALL_TOOLS}
    if unknown:
        raise ValueError(f"Unknown tools specified in task: {sorted(unknown)}")
    return [t for t in ALL_TOOLS if t.__name__ in wanted]


def build_world(info: dict) -> Any:
    """The sample's ``WorldState``, built as the official runner builds it: from the
    ``None``-stripped ``initial_state``, subscribed to the services the task seeds, asserts on
    or grants tools for."""
    from automationbench.runner import WorldState, compute_allowed_services, strip_none_values

    initial_state = strip_none_values(info.get("initial_state", {}))
    world = WorldState(**initial_state)
    world.meta.allowed_services = compute_allowed_services(
        initial_state, strip_none_values(info.get("assertions", [])), info.get("zapier_tools", []))
    return world


@solver
def automationbench_agent(toolset: str = "api", max_turns: int = MAX_TURNS) -> Solver:
    """Run the AutomationBench loop for one sample (see the module docstring).

    ``toolset`` is ``api``, ``zapier`` or ``limited_zapier``; ``max_turns`` caps the model's
    replies. ``AutomationBench`` must be installed (``tasks/agentic/automationbench.py`` checks).
    """
    if toolset not in TOOLSETS:
        raise ValueError(f"toolset must be one of {TOOLSETS}, got {toolset!r}")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if toolset == "api":
        # api_search caches its index in the package directory on first use; building it here
        # keeps concurrent samples from racing to write it.
        from automationbench.tools.api import api_search
        api_search("index", top_k=1)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        info = state.metadata[INFO_KEY]
        world = build_world(info)
        defs = [_tool_def(f, world) for f in _tool_functions(toolset, info)]
        state.tools = [d.as_tool() for d in defs]

        turns, limit = 0, None
        while True:
            state = await generate(state, tool_calls="none")
            turns += 1
            if state.output.stop_reason == "model_length":
                limit = "context"
                break
            if not state.output.message.tool_calls:
                break
            if turns >= max_turns:
                limit = "turns"
                break
            result = await execute_tools(state.messages, state.tools, max_output=NO_TRUNCATION)
            state.messages.extend(result.messages)

        state.store.set(WORLD_KEY, world.model_dump(mode="json"))
        state.store.set(LIMIT_REACHED, limit)
        state.store.set(TURNS, turns)
        return state

    return solve
