"""DeepSWE: long-horizon feature work in real open-source repositories, graded by held-out tests.

Official source: DeepSWE by Datacurve (https://github.com/datacurve-ai/deep-swe, Apache-2.0
for Datacurve's own material; each task's upstream repository keeps its own licence, listed in
the repository's ``PROVENANCE.md``). 113 tasks (TypeScript, Go, Python, JavaScript, Rust) in the
Harbor task format, pinned here to ``datacurve/deep-swe-1-1`` at a dataset digest (decision
records 0008, 0020). The reward is 1 when the held-out tests pass and 0 when not (the verifier's
``reward.json`` also holds pass fractions, kept in ``Score.metadata["reward_dict"]``); accuracy is
the share of tasks resolved. A grader that crashes writes ``-1``; that sample is an error, not a 0.

Followed from the benchmark: the tasks, containers and tests; mini-swe-agent as the agent loop
(the leaderboard's) with its built-in configuration, no step limit and no cost limit, and mini's
30-second shell command timeout; the per-task agent time limit (``[agent].timeout_sec`` of
``task.toml``), applied here because ``inspect_harbor`` ignores it (the run is stopped at the
limit and scored on its final state, counted by ``agent_diagnostics``).

Departures:

- **Verifier.** Since version 1.1 DeepSWE grades in a separate container: the agent commits its
  work, a ``[[verifier.collect]]`` hook extracts the commits as a patch, and the patch is applied
  and graded in a pristine container. ``inspect_harbor`` runs the hook, then resets the
  repository to its base commit in the agent's own container and grades there. Build products
  or edits to tracked files that the agent did not commit are lost, and anything the agent
  left outside the repository stays. Scores can differ from the official ones for that reason.
- **Agent loop.** The leaderboard runs mini-swe-agent through Pier, a Harbor-compatible runner;
  here it runs through ``inspect_swe`` (``max_steps`` 0, no limit, by default). ``react`` and
  the vendor command-line agents are options (see ``agent_solver``). The log records the
  scaffold; compare only runs with the same one.
- **Sandbox.** Containers come from the task's environment definition, through a
  Docker-compatible provider chosen by ``sandbox`` (``docker`` by default; ``inspect eval
  --sandbox`` overrides it). Memory is raised to at least 6 GB when a task asks for less.
  The leaderboard uses Modal, and per-agent network allowlists that a plain container does not
  have.
- **Repeats.** One run (``epochs`` 1).
- **Reasoning-model handling** (decision record 0020): thinking on, sampling from the served
  model's ``generation_config``, no ``max_tokens``.

DeepSWE's evaluation data is gated on Hugging Face under terms for evaluation use only; the task
archives this task downloads come from the Harbor hub. Accept the dataset terms on Hugging
Face before use and keep to them (see ``docs/benchmarks.md``).

Needs ``pip install "dna-eval[agentic]"`` and Python 3.12 or later; the model server needs tool
calling (``docs/agentic.md``). Run with ``--no-fail-on-error --retry-on-error 3``: a sandbox or
verifier that breaks is a sample error, not a score of 0.
"""
from __future__ import annotations

from inspect_ai import Task, task

from dna_eval.tasks.agentic.harbor import harbor_task

PACKAGE = "datacurve/deep-swe-1-1"
DATASET_DIGEST = "sha256:5affcd534fd90ac85d202d4c63f8b35ddc942140afdd5d60014a21365440a2f5"   # hub revision 1, 113 tasks


@task
def deep_swe(scaffold: str = "mini_swe_agent", thinking: bool = True, epochs: int = 1, max_steps: int = 0,
             command_timeout: int | None = None, sandbox: str = "docker",
             task_names: str | list[str] | None = None, n_tasks: int | None = None) -> Task:
    """DeepSWE 1.1 (113 tasks), one run by default. Headline: ``accuracy``, the share of tasks
    resolved.

    ``scaffold``: ``mini_swe_agent``, ``react``, ``claude_code`` or ``codex_cli``.
    ``max_steps``: model turns allowed per run (0: no limit, as the leaderboard).
    ``command_timeout``: seconds a shell command may run in mini-swe-agent (``None``: 30, its default).
    ``sandbox``: the Docker-compatible sandbox provider. ``task_names`` and ``n_tasks`` select a
    subset for a smoke test. Reports ``accuracy``, ``stderr`` and ``agent_diagnostics``.
    """
    return harbor_task(
        package=PACKAGE, digest=DATASET_DIGEST, scaffold=scaffold, max_steps=max_steps,
        command_timeout=command_timeout, epochs=epochs, thinking=thinking, sandbox_type=sandbox,
        task_names=task_names, n_tasks=n_tasks)
