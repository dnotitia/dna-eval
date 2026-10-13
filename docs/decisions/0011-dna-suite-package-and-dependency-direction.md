# 0011. The DNA suite is a package; dependencies point one way

- Status: accepted
- Date: 2026-10-12

## Context

Suite-wide conventions (generation settings, headline metric) were spread over modules, and a
scorer imported a solver to read a value the solver had stored.

## Decision

- The suite's tasks live in `tasks/dna_suite/`. Its `__init__.py` holds the suite conventions
  (`dna_generate_config`, `MAX_TOKENS`, `HEADLINE`) and imports no task module.
- Dependencies point down: `tasks` → `solvers` / `scorers` / `metrics` → `judging` / `prompts` /
  `rubrics` / `dataset` → `utils`. Solvers and scorers do not import each other or `tasks`.
- A value a solver hands to the judge travels as data: the solver writes `state.store` or
  `state.metadata`, and the rubric's `sources` names the key (`store.reasoning`).

## Consequences

Reusable pieces (solvers, scorers) carry no suite assumptions in their imports; the suite's
choices are visible in one module and in the rubric files.
