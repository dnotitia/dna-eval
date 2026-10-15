"""Terminal-Bench: terminal tasks an agent solves in a container, graded by each task's tests.

Official source: Terminal-Bench (https://github.com/harbor-framework/terminal-bench, Apache-2.0),
published as datasets on the Harbor hub and run by Harbor (https://harborframework.com). A task
is a directory: ``instruction.md`` (the prompt), ``environment/`` (the container), ``tests/`` (the
grader) and ``task.toml`` (resources and the agent and verifier time limits). The agent works in
the container; afterwards the task's test script runs there and writes a reward, 1 when the
tests pass and 0 when not. Accuracy is the mean reward (Terminal-Bench's resolve rate).

Versions, pinned by the digest of the dataset on the hub (decision records 0008, 0020):

- ``"2.1"``: ``terminal-bench/terminal-bench-2-1``, 89 tasks. The default.
- ``"4.0"``: ``terminal-bench/terminal-bench``, 66 tasks, **experimental**. Version 4.0 grades
  in a separate verifier container; the dependency (``inspect_harbor``) does not support that
  yet, so the tests run in the agent's container, after the agent, and a task whose tests
  depend on a clean container can score 0 or fail for a reason that has nothing to do with the
  agent. Compare these numbers with nothing. A warning is logged on every run. Three of its
  tasks need a GPU, so they need a sandbox provider that offers one.

Followed from the benchmark: the tasks, their containers and resource requests, their tests
and reward; the per-task agent time limit (``[agent].timeout_sec`` of ``task.toml``), applied
here because ``inspect_harbor`` ignores it (the run is stopped at the limit and scored on its
final state, counted by ``agent_diagnostics``); the per-task verifier time limit.

Departures:

- **Scaffold.** The reference agent is Terminus 2, run by Harbor. Here the agent loop is
  ``scaffold``: ``mini_swe_agent`` (default; mini-swe-agent, a bash-only loop, installed in
  the container with pip, which needs network access and Python 3.11 or ``uv`` able to fetch
  it), ``react`` (Inspect's ReAct agent with bash, python and plan tools) or one of the vendor
  command-line agents (see ``agent_solver``). Scores depend on the scaffold, which the log
  records; compare only runs with the same one.
- **Turn limit.** The benchmark sets no turn limit, only the time limit. ``max_steps`` caps
  model turns as a safeguard against a loop that never ends: ``None`` uses 0 (no cap) for
  version 2.1 and 500 for version 4.0; ``0`` means no cap. A run a cap ends is scored on its
  final state.
- **Shell command timeout.** mini-swe-agent ends a shell command after 30 seconds, its own
  default, and tells the model; a command that needs longer, such as a build, fails for the
  agent. Raise it with ``command_timeout`` if that is not wanted; the ``react`` scaffold gives
  each tool call 300 seconds.
- **Sandbox.** Containers come from the task's environment definition, through a
  Docker-compatible provider chosen by ``sandbox`` (``docker`` by default; ``inspect eval
  --sandbox`` overrides it). Memory is raised to at least 6 GB when a task asks for less
  (``inspect_harbor``'s floor). Tasks with an egress allowlist run with open network access.
- **Repeats.** ``epochs`` defaults to 5: the Terminal-Bench leaderboard asks for at least
  five trials per task. Accuracy is the mean over runs.
- **Reasoning-model handling** (decision record 0020): thinking on, sampling from the served
  model's ``generation_config``, no ``max_tokens``.

Needs ``pip install "dna-eval[agentic]"`` and Python 3.12 or later; the model server needs tool
calling (``docs/agentic.md``). Run with ``--no-fail-on-error --retry-on-error 3``: a sandbox or
verifier that breaks is a sample error, not a score of 0.
"""
from __future__ import annotations

import logging

from inspect_ai import Task, task

from dna_eval.tasks.agentic.harbor import harbor_task

logger = logging.getLogger(__name__)

PACKAGE = "terminal-bench/terminal-bench-2-1"
DATASET_DIGEST = "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"   # hub revision 6, 89 tasks
PACKAGE_4 = "terminal-bench/terminal-bench"
DATASET_DIGEST_4 = "sha256:39d9f44b40420cde8fdcc087579c0d72a7e14fa3656d603c3f0d22fb35e27732"   # hub revision 4 (4.0.0), 66 tasks

# version -> (package, digest, default max_steps)
VERSIONS = {"2.1": (PACKAGE, DATASET_DIGEST, 0), "4.0": (PACKAGE_4, DATASET_DIGEST_4, 500)}

EXPERIMENTAL_4 = ("Terminal-Bench 4.0 is experimental here: its separate verifier container is not "
                  "supported by inspect_harbor, so the tests run in the agent's container and some tasks "
                  "can score 0 or error for reasons unrelated to the agent")


@task
def terminal_bench(version: str = "2.1", scaffold: str = "mini_swe_agent", thinking: bool = True, epochs: int = 5,
                   max_steps: int | None = None, command_timeout: int | None = None, sandbox: str = "docker",
                   task_names: str | list[str] | None = None, n_tasks: int | None = None) -> Task:
    """Terminal-Bench ``version`` ("2.1": 89 tasks; "4.0": 66 tasks, experimental), ``epochs`` runs
    each. Headline: ``accuracy``, the mean reward.

    ``scaffold``: ``mini_swe_agent``, ``react``, ``claude_code`` or ``codex_cli``.
    ``max_steps``: model turns allowed per run (``None``: 0, no cap, for 2.1; 500 for 4.0).
    ``command_timeout``: seconds a shell command may run in mini-swe-agent (``None``: 30).
    ``sandbox``: the Docker-compatible sandbox provider. ``task_names`` and ``n_tasks`` select a
    subset for a smoke test. Reports ``accuracy``, ``stderr`` and ``agent_diagnostics``.
    """
    if version not in VERSIONS:
        raise ValueError(f"unknown Terminal-Bench version {version!r}; choose one of {', '.join(VERSIONS)}")
    package, digest, default_steps = VERSIONS[version]
    if version == "4.0":
        logger.warning(EXPERIMENTAL_4)
    return harbor_task(
        package=package, digest=digest, scaffold=scaffold,
        max_steps=default_steps if max_steps is None else max_steps, command_timeout=command_timeout,
        epochs=epochs, thinking=thinking, sandbox_type=sandbox, task_names=task_names, n_tasks=n_tasks,
        metadata={"version": version, **({"experimental": EXPERIMENTAL_4} if version == "4.0" else {})})
