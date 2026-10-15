# 0020. Agent tasks share one set of conventions; the environment is the user's choice

- Status: accepted
- Date: 2026-10-14
- Scope: tasks in `tasks/agentic/` (tau-bench, AutomationBench, Terminal-Bench, SWE-bench,
  DeepSWE, ...): the model under evaluation acts in a loop with tools, sometimes with a
  simulated user.

## Context

Agent benchmarks score a model through an environment: simulated services, a container, a
simulated user. Their official code defines that environment and its grading; their scores
depend on the agent loop (scaffold), the limits, and the model playing the user. dna-eval is a
framework others run on their own infrastructure.

## Decision

- **Official environment and grading, as dependencies.** A task calls the benchmark's own
  environment and evaluator code, imported inside the task, solver or scorer function. A
  package on PyPI goes in the `agentic` extra; one that is not is installed by the command the
  task's error message and `docs/agentic.md` give (`tasks/agentic/require`).
- **The environment is the user's choice.** A task that needs a container declares a
  Docker-compatible sandbox; the provider (`docker` by default, `k8s`, cloud providers) is
  chosen at run time. dna-eval depends on no provider.
- **Scaffold and limits are explicit.** Where the agent loop is not the benchmark's own, it is a
  task argument with the official or most-cited setup as default, recorded in the log; a
  substitute is a documented deviation. Turn, step and time limits follow the official settings.
  A run a limit ends is scored on its final state and counted by `agent_diagnostics`.
- **Infrastructure failures are errors.** A sandbox or verifier that breaks makes the sample an
  error (retried with `--retry-on-error`), not a score of 0.
- **Roles.** A simulated user is the `user` role and a judge the `grader` role, both required
  (0010); the models bound decide comparability and are recorded in the log.
- **Pinned data** (0008): Hub datasets by revision, Harbor datasets by `sha256` digest, files in
  git repositories by commit and checksum.
- **Reasoning models** as in 0016: thinking on, sampling from `generation_config`, no
  `max_tokens`.

## Consequences

Numbers depend on the scaffold, the user and judge models and the benchmark version, all in
the log. Some tasks need Python 3.12 or 3.13 and packages from git. Benchmarks whose official
grading the dependencies cannot yet run (Terminal-Bench 4.0's separate verifier) are marked
experimental rather than approximated silently.

## Alternatives considered

- Reimplementing environments in dna-eval: diverges from the official grading.
- Adopting an environment protocol (OpenEnv, verifiers, OpenReward): an extra translation layer,
  unstable APIs or hosted services, and no better grading fidelity.
- Fixing one sandbox provider: ties the framework to one kind of infrastructure.
