"""``agent_solver``: run an agent loop (scaffold) in the sample's sandbox.

A benchmark whose official numbers come from an agent of the reporter's choice (Terminal-Bench,
SWE-bench, DeepSWE) is run by one of the scaffolds below, chosen by the task (decision record
0020). ``inspect_swe`` is imported inside the functions that use it, so this module imports
without it.
"""
from __future__ import annotations

import yaml
from inspect_ai.agent import Agent, AgentState
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import apply_limits, sandbox, time_limit, turn_limit

from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS
from dna_eval.utils.optional import AGENTIC_INSTALL, require

# Scaffolds ``agent_solver`` can run; ``mini_swe_agent`` is the default of the Harbor tasks.
SCAFFOLDS = ("mini_swe_agent", "react", "claude_code", "codex_cli")

# mini-swe-agent release run inside the sandbox. ``inspect_swe`` installs it there with pip and
# its default is whatever ``inspect_swe`` pins; pinning it here keeps the prompts below, which
# are this release's, in step with the program that reads them.
MINI_SWE_AGENT_VERSION = "2.2.3"

# Where the configuration below is written inside the sandbox; ``mini`` reads it through
# ``MSWEA_MINI_CONFIG_PATH`` in place of its built-in ``mini.yaml``.
MINI_SWE_AGENT_CONFIG_PATH = "/var/tmp/dna-eval-mini-swe-agent.yaml"

# Environment of the ``mini`` process: its litellm layer prices every call by model name and
# stops on an unknown name, which is every self-hosted model, unless told to ignore cost
# errors; the local price table avoids a download from inside the sandbox.
MINI_SWE_AGENT_ENV = {"MSWEA_COST_TRACKING": "ignore_errors", "LITELLM_LOCAL_MODEL_COST_MAP": "true"}

# Timeout, in seconds, of one tool call of the ``react`` scaffold (inspect_harbor's own value).
REACT_TOOL_TIMEOUT = 300

# Third-party text: SWE-agent/mini-swe-agent 2.2.3 (MIT, Copyright (c) 2025 Kilian A. Lieret and
# Carlos E. Jimenez); see THIRD_PARTY_NOTICES.md. ``src/minisweagent/config/mini.yaml``, verbatim:
# the system and instance prompts, the observation and format-error templates and the shell
# environment of the agent. ``mini_swe_agent_config`` changes only the step limit, the cost limit
# and, when asked, the command timeout.
MINI_SWE_AGENT_YAML = r'''
agent:
  system_template: |
    You are a helpful assistant that can interact with a computer.
  instance_template: |
    Please solve this issue: {{task}}

    You can execute bash commands and edit files to implement the necessary changes.

    ## Recommended Workflow

    This workflows should be done step-by-step so that you can iterate on your changes and any possible problems.

    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust
    6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
       Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

    ## Command Execution Rules

    You are operating in an environment where

    1. You issue at least one command
    2. The system executes the command(s) in a subshell
    3. You see the result(s)
    4. You write your next command(s)

    Each response should include:

    1. **Reasoning text** where you explain your analysis and plan
    2. At least one tool call with your command

    **CRITICAL REQUIREMENTS:**

    - Your response SHOULD include reasoning text explaining what you're doing
    - Your response MUST include AT LEAST ONE bash tool call
    - Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
    - However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
    - Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
      Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

    Example of a CORRECT response:
    <example_response>
    I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

    [Makes bash tool call with {"command": "ls -la"} as arguments]
    </example_response>

    <system_information>
    {{system}} {{release}} {{version}} {{machine}}
    </system_information>

    ## Useful command examples

    ### Create a new file:

    ```bash
    cat <<'EOF' > newfile.py
    import numpy as np
    hello = "world"
    print(hello)
    EOF
    ```

    ### Edit files with sed:

    {%- if system == "Darwin" -%}
    <important>
    You are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.
    </important>
    {%- endif -%}

    ```bash
    # Replace all occurrences
    sed -i 's/old_string/new_string/g' filename.py

    # Replace only first occurrence
    sed -i 's/old_string/new_string/' filename.py

    # Replace first occurrence on line 1
    sed -i '1s/old_string/new_string/' filename.py

    # Replace all occurrences in lines 1-10
    sed -i '1,10s/old_string/new_string/g' filename.py
    ```

    ### View file content:

    ```bash
    # View specific lines with numbers
    nl -ba filename.py | sed -n '10,20p'
    ```

    ### Any other command you want to run

    ```bash
    anything
    ```
  step_limit: 0
  cost_limit: 3.
  mode: confirm
environment:
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: 'off'
    TQDM_DISABLE: '1'
model:
  observation_template: |
    {%- if output.output | length < 10000 -%}
    {
      "returncode": {{ output.returncode }},
      "output": {{ output.output | tojson }}
      {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
    }
    {%- else -%}
    {
      "returncode": {{ output.returncode }},
      "output_head": {{ output.output[:5000] | tojson }},
      "output_tail": {{ output.output[-5000:] | tojson }},
      "elided_chars": {{ output.output | length - 10000 }},
      "warning": "Output too long."
      {%- if output.exception_info %}, "exception_info": {{ output.exception_info | tojson }}{% endif %}
    }
    {%- endif -%}
  format_error_template: |
    Tool call error:

    <error>
    {{error}}
    </error>

    Here is general guidance on how to submit correct toolcalls:

    Every response needs to use the 'bash' tool at least once to execute commands.

    Call the bash tool with your command as the argument:
    - Tool: bash
    - Arguments: {"command": "your_command_here"}

    If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
    without any other command.
  model_kwargs:
    drop_params: true
'''


def mini_swe_agent_config(step_limit: int = 0, command_timeout: int | None = None) -> str:
    """The mini-swe-agent configuration file (YAML text) the ``mini_swe_agent`` scaffold writes
    into the sandbox: mini's built-in ``mini.yaml`` with ``agent.step_limit`` set to
    ``step_limit`` (0 = no limit), ``agent.cost_limit`` 0 (no limit: the dollar cap is
    meaningless for a model with no price) and, if ``command_timeout`` is given,
    ``environment.timeout`` (seconds one shell command may run; mini's own default is 30)."""
    config = yaml.safe_load(MINI_SWE_AGENT_YAML)
    config["agent"]["step_limit"] = step_limit
    config["agent"]["cost_limit"] = 0
    if command_timeout is not None:
        config.setdefault("environment", {})["timeout"] = command_timeout
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


async def _setup_agent(state: TaskState, scaffold: str, step_limit: int, command_timeout: int | None) -> Agent:
    """The agent for one sample. It runs as the task's ``[agent].user`` when the task sets one
    (``metadata["agent_user"]``); ``mini_swe_agent`` gets its configuration file written into
    the sandbox first."""
    user = state.metadata.get("agent_user")
    if scaffold == "react":
        from inspect_ai.agent import react
        from inspect_ai.model import CompactionEdit
        from inspect_ai.tool import bash, python, update_plan

        return react(tools=[bash(timeout=REACT_TOOL_TIMEOUT, user=user),
                            python(timeout=REACT_TOOL_TIMEOUT, user=user), update_plan()],
                     compaction=CompactionEdit())
    require("inspect_swe", install=AGENTIC_INSTALL)
    import inspect_swe

    if scaffold == "mini_swe_agent":
        await sandbox().write_file(MINI_SWE_AGENT_CONFIG_PATH, mini_swe_agent_config(step_limit, command_timeout))
        return inspect_swe.mini_swe_agent(
            version=MINI_SWE_AGENT_VERSION, user=user,
            env={**MINI_SWE_AGENT_ENV, "MSWEA_MINI_CONFIG_PATH": MINI_SWE_AGENT_CONFIG_PATH})
    if scaffold == "claude_code":
        return inspect_swe.claude_code(user=user)
    return inspect_swe.codex_cli(user=user)


async def _run_agent(agent: Agent, state: TaskState, *, max_turns: int, timeout_sec: float | None,
                     native_turn_limit: bool = False) -> None:
    """Run ``agent`` on ``state`` under a turn limit and a wall-clock limit.

    A limit that ends the run does not fail the sample: the run is scored on its final state,
    and ``state.store`` records which limit ended it (``LIMIT_REACHED``: ``"time"`` or
    ``"turns"``) and how many model turns were taken (``TURNS``). ``max_turns`` 0 and
    ``timeout_sec`` ``None`` set no limit. ``native_turn_limit`` says the agent enforces
    ``max_turns`` itself and stops cleanly (mini-swe-agent does): a run that used up its turns
    is then counted as ended by the turn limit.
    """
    turns = turn_limit(max_turns or None)
    limits = [turns] + ([time_limit(timeout_sec)] if timeout_sec else [])
    agent_state = AgentState(messages=state.messages)
    # An exception other than a limit of ours still ends the sample, as the agent solver does
    # for any agent: the messages so far are kept in the log and are what is scored.
    try:
        with apply_limits(limits, catch_errors=True) as scope:
            agent_state = await agent(agent_state)
    finally:
        state.messages = agent_state.messages
        if agent_state.output:
            state.output = agent_state.output
    used = int(turns.usage) or sum(isinstance(m, ChatMessageAssistant) for m in state.messages)
    limit_error = scope.limit_error
    if limit_error is not None:
        reached = {"time": "time", "turn": "turns"}.get(limit_error.type, limit_error.type)
        if limit_error.type == "turn" and max_turns:
            used = min(used, max_turns)     # the call that tripped the limit was counted but its reply dropped
    else:
        reached = "turns" if native_turn_limit and max_turns and used >= max_turns else None
    state.store.set(LIMIT_REACHED, reached)
    state.store.set(TURNS, used)


@solver
def agent_solver(scaffold: str = "mini_swe_agent", max_steps: int = 0, command_timeout: int | None = None) -> Solver:
    """Run an agent in the sample's sandbox: the loop that stands in for a benchmark's own agent.

    ``scaffold``:

    - ``mini_swe_agent``: mini-swe-agent (``inspect_swe``) with its built-in prompts, a bash-only
      loop that submits by printing a marker. ``max_steps`` is its ``step_limit``; the cost limit
      is off. See ``mini_swe_agent_config``.
    - ``react``: Inspect's ReAct agent with the tools ``inspect_harbor`` gives it (``bash``,
      ``python``, ``update_plan``) and edit compaction; ``max_steps`` is a limit on model turns.
    - ``claude_code``, ``codex_cli``: the vendors' command-line agents through ``inspect_swe``;
      ``max_steps`` is a host-side limit on model turns. These two are not exercised by the
      offline tests.

    ``max_steps`` is the most model turns the agent may take (0 = no limit). The sample's
    ``metadata["harbor_config"]["agent"]["timeout_sec"]``, the task's own agent time limit,
    ends the run when it passes (``inspect_harbor`` does not apply it). A run a limit ends is
    scored on its final state; ``state.store`` gets ``LIMIT_REACHED`` and ``TURNS`` for
    ``agent_diagnostics``. ``command_timeout`` is mini-swe-agent's seconds per shell command
    (``None``: its default, 30).
    """
    if scaffold not in SCAFFOLDS:
        raise ValueError(f"unknown scaffold {scaffold!r}; choose one of {', '.join(SCAFFOLDS)}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        agent_config = (state.metadata.get("harbor_config") or {}).get("agent") or {}
        agent = await _setup_agent(state, scaffold, max_steps, command_timeout)
        await _run_agent(agent, state, max_turns=max_steps, timeout_sec=agent_config.get("timeout_sec"),
                         native_turn_limit=scaffold == "mini_swe_agent")
        return state

    return solve
