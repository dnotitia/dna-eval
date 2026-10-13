# 0001. Build on Inspect AI; DeepEval parity covers only the ported DNA suite

- Status: accepted
- Date: 2026-10-12

## Context

dna-eval is built on Inspect AI. The DNA suite (`performance`, `uncensorship`, `persona`,
`hallucination`, `degeneration`, `style`) was first run with a DeepEval GEval harness, and its
published numbers must stay comparable after the port. Tasks added later have no such history.

## Decision

- dna-eval is a layer on Inspect: tasks, solvers, scorers and metrics are Inspect objects, and
  Inspect's own mechanisms (model roles, the registry, logs, headline metrics) come first. We
  add our own only where Inspect falls short, and record why.
- **Parity is scoped to the ported suite.** For the tasks in `tasks/dna_suite/` ported from the
  DeepEval harness, every behaviour that moves a score reproduces DeepEval GEval and is pinned
  by a test under `tests/parity/`; a deviation is a recorded decision.
- Every other task, whether new or added to the suite later, does not need DeepEval parity.
  It follows Inspect's and this project's conventions and is checked by its own tests. It may
  reuse the suite's building blocks (`rubric_scorer`, `dna_generate`) or not.

## Consequences

Records 0004–0007 describe how the ported suite stays comparable; they constrain new tasks only
where those tasks choose the same building blocks. The parity tests are a migration safeguard:
once the old numbers no longer need comparing, a superseding record can retire them.

## Alternatives considered

- DeepEval parity for every task: ties tasks to a harness they never ran on, and blocks better
  choices (another judge prompt, other metrics) for no comparability gain.
