# 0017. Third-party evaluation code is vendored unchanged, with its licence

- Status: superseded by 0021
- Date: 2026-10-12
- Scope: code copied from benchmark authors into `src/dna_eval/vendor/`.

## Context

Some benchmarks are defined by their checking code, not only by a prompt and a regex: the
IFEval and IFBench verifiers are about 80 hand-written checks. Reimplementing them would make
the numbers depend on our reading of each check; the upstream packages are not published on
PyPI in a form that installs cleanly, and some import NLTK data at import time.

## Decision

- Copy the upstream code into `src/dna_eval/vendor/<name>/`, with the upstream `LICENSE` and a
  `NOTICE.md` naming the source repository, the commit, the files copied, and every local change.
- Keep local changes to what packaging needs (package-relative imports, no downloads or file
  writes at import time). Checking logic is not edited; a known upstream quirk is documented,
  not fixed, so results stay comparable with the upstream scripts.
- Only licences that allow redistribution with notice (Apache-2.0, MIT, BSD) are vendored.
- Code that calls the vendored package (scorers, metrics) lives outside `vendor/`; nothing in
  `vendor/` imports from the rest of `dna_eval`.

## Consequences

Updating a vendored package is a deliberate change: a new commit in `NOTICE.md`, the local
changes re-applied, and the scores it produces re-checked. The vendored packages' runtime
dependencies are dependencies of `dna-eval`.

## Alternatives considered

- Depend on the upstream repository from git: unpinned in practice, and its import-time
  downloads would run on every `inspect` start.
- Reimplement the checks: our interpretation would become part of every number.
