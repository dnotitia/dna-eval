"""SWE-bench Verified and SWE-bench Pro: fix a real repository issue, graded by the tests.

Official source: SWE-bench (https://github.com/SWE-bench/SWE-bench, MIT) for Verified and Scale
AI's SWE-bench Pro (https://github.com/scaleapi/SWE-bench_Pro-os) for Pro, both as Harbor
datasets on the Harbor hub. Each task is a repository checked out at a base commit inside its
container, an issue as the prompt, and tests that must pass after the agent's edits (the
tests are copied in after the agent finishes). The reward is 1 when they pass and 0 when not;
accuracy is the share of tasks resolved.

Variants, pinned by the digest of the dataset on the hub (decision records 0008, 0020):

- ``"pro"``: ``scale-ai/swe-bench-pro``, 731 tasks. The default.
- ``"pro_hardened"``: ``cais/swebenchpro``, the same 731 tasks as published by CAIS, described
  as hardened against agents that exploit the grader.
- ``"verified"``: ``swe-bench/swe-bench-verified``, the 500 human-validated SWE-bench tasks.

Followed from the benchmark: the tasks, containers and tests; the per-task agent time limit
(``[agent].timeout_sec`` of ``task.toml``), applied here because ``inspect_harbor`` ignores it
(the run is stopped at the limit and scored on its final state); the per-task verifier limit.

Departures:

- **Scaffold.** The official SWE-bench setting for comparing models with a bash-only loop is
  mini-swe-agent with ``step_limit`` 250 and a cost limit of 3 dollars; that is the default
  here (``scaffold="mini_swe_agent"``, ``max_steps=250``). Differences: the cost limit is off
  (a self-hosted model has no price; the 250-step limit still applies); the prompt is
  mini-swe-agent's general ``mini.yaml`` prompt, not its SWE-bench one, which names
  ``/testbed``, the working directory of Verified only (Pro images use ``/app``); a shell
  command may run 60 seconds, the SWE-bench setting (``command_timeout``). The step limit is
  enforced inside mini-swe-agent, which ends the run cleanly. ``react`` (Inspect's ReAct agent)
  and the vendor command-line agents are options (see ``agent_solver``); Pro's own reference
  numbers use Scale AI's agent loop, not this one. The log records the scaffold; compare only
  runs with the same one.
- **Verifier.** The tests run in the agent's container after it finishes, as ``inspect_harbor``
  does for every Harbor task.
- **Sandbox.** Containers come from the task's environment definition, through a
  Docker-compatible provider chosen by ``sandbox`` (``docker`` by default; ``inspect eval
  --sandbox`` overrides it). Memory is raised to at least 6 GB when a task asks for less.
- **Repeats.** One run (``epochs`` 1): pass@1, as published.
- **Reasoning-model handling** (decision record 0020): thinking on, sampling from the served
  model's ``generation_config``, no ``max_tokens``.

Data licences differ by task: the SWE-bench repositories are under their own open-source
licences, and the Pro set includes repositories under the GPL; see ``docs/benchmarks.md``.

Needs ``pip install "dna-eval[agentic]"`` and Python 3.12 or later; the model server needs tool
calling (``docs/agentic.md``). Run with ``--no-fail-on-error --retry-on-error 3``: a sandbox or
verifier that breaks is a sample error, not a score of 0.
"""
from __future__ import annotations

from inspect_ai import Task, task

from dna_eval.tasks.agentic.harbor import harbor_task

# variant -> (hub package, digest of the dataset version this task is pinned to)
VARIANTS = {
    "pro": ("scale-ai/swe-bench-pro",
            "sha256:88411d32ff27e53a4c1a7e29f0c2aeba180c8e5d60f221cab5ed56325f33549d"),        # revision 2, 731 tasks
    "pro_hardened": ("cais/swebenchpro",
                     "sha256:0684038ce8eae92d435a27307d1c5843e291152898f429af130062e8df110768"),  # revision 2, 731 tasks
    "verified": ("swe-bench/swe-bench-verified",
                 "sha256:b934b0cc3dc800fe945eaf9f1623329db97ee3133c706d20644524c7759fb341"),   # revision 2, 500 tasks
}

STEP_LIMIT = 250        # mini-swe-agent's SWE-bench step limit
COMMAND_TIMEOUT = 60    # seconds a shell command may run, as in mini-swe-agent's SWE-bench configuration


@task
def swe_bench(variant: str = "pro", scaffold: str = "mini_swe_agent", thinking: bool = True, epochs: int = 1,
              max_steps: int | None = None, command_timeout: int | None = None, sandbox: str = "docker",
              task_names: str | list[str] | None = None, n_tasks: int | None = None) -> Task:
    """SWE-bench ``variant`` ("pro": 731 tasks; "pro_hardened": 731; "verified": 500), one run by
    default. Headline: ``accuracy``, the share of tasks resolved.

    ``scaffold``: ``mini_swe_agent``, ``react``, ``claude_code`` or ``codex_cli``.
    ``max_steps``: model turns allowed per run (``None``: 250; 0: no limit).
    ``command_timeout``: seconds a shell command may run in mini-swe-agent (``None``: 60).
    ``sandbox``: the Docker-compatible sandbox provider. ``task_names`` and ``n_tasks`` select a
    subset for a smoke test. Reports ``accuracy``, ``stderr`` and ``agent_diagnostics``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown SWE-bench variant {variant!r}; choose one of {', '.join(VARIANTS)}")
    package, digest = VARIANTS[variant]
    return harbor_task(
        package=package, digest=digest, scaffold=scaffold,
        max_steps=STEP_LIMIT if max_steps is None else max_steps,
        command_timeout=COMMAND_TIMEOUT if command_timeout is None else command_timeout,
        epochs=epochs, thinking=thinking, sandbox_type=sandbox, task_names=task_names, n_tasks=n_tasks,
        metadata={"variant": variant})
