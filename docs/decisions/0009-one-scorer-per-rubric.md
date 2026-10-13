# 0009. One registered scorer per rubric

- Status: accepted
- Date: 2026-10-12

## Context

With one generic `@scorer`, Inspect labels results `rubric_scorer`, `rubric_scorer1`, ...;
the log viewer and `inspect score` need a scorer's name to say what it measures.

## Decision

- Importing `scorers/rubric_scorer.py` registers every rubric file as a scorer of the same name
  (`dna_eval/performance`, `dna_eval/uncensorship`, ...). `rubric_scorer(name)` is the single
  entry point; it is a plain function, not itself registered.
- `_registry.py` (the package entry point) imports every module that registers objects, so all
  of them resolve by name whether or not a task uses them.
- Re-score a log with `inspect score <log> --scorer dna_eval/<rubric> --action overwrite|append`.

## Consequences

Re-scoring needs the namespaced name: logs record unqualified names, and Inspect does not load
a package for an unqualified name. `--action overwrite` replaces all scores, not only that scorer's.

## Alternatives considered

- Generic `@scorer rubric_scorer(rubric=...)`: unreadable result names.
- Naming scorers at task construction time: the names did not exist in a fresh process, so logs
  could not be re-scored.
