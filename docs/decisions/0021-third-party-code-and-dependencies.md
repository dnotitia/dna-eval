# 0021. Third-party code is depended on, copied where used, or vendored; what moves scores is pinned

- Status: accepted
- Date: 2026-10-15
- Scope: every piece of code, text or data from outside dna-eval that a task, solver, scorer or
  metric runs. Supersedes 0017.

## Context

Benchmarks bring their authors' code: prompts and answer patterns, verifiers, evaluators,
whole environments. dna-eval used three ways to take it in (a copy beside its use, a vendored
copy, a dependency) without saying when each applies, and pinned it unevenly: data and
vendored code to a commit, some dependencies to an exact version, others to a lower bound
only. Anything that grades a benchmark defines its scores; an unpinned update changes them
without anyone deciding to.

## Decision

- **Choose in this order.**
  1. *Dependency*: the upstream package is called as it is, through its API (an environment,
     an evaluator, a harness). Imported inside the function that needs it when only some tasks
     use it; a package on PyPI goes in an extra, one that is not is installed by the command
     the task's `require` error and its guide give.
  2. *Copy where used*: a short text or function (a prompt, an answer pattern, one or two
     functions) used by one module, with a source comment.
  3. *Vendor*: several upstream files that must run unchanged as a unit and cannot be a
     dependency (not installable as published, side effects at import, or run inside a
     sandbox). They go in `src/dna_eval/vendor/<name>/` with the upstream `LICENSE` and a
     `NOTICE.md` naming the repository, the commit, the files and every local change; local
     changes are limited to packaging, and nothing in `vendor/` imports the rest of `dna_eval`.
- **Licences.** Only licences that allow redistribution with notice are copied or vendored;
  every copy is listed in `THIRD_PARTY_NOTICES.md` (0018).
- **Pinning.** What decides a score is pinned: data (0008), vendored and copied code (by
  commit), and the dependencies that grade or simulate (`==` or a compatible-release range
  `~=`); a sandbox image pins its packages in its Dockerfile. Infrastructure libraries keep a
  lower bound.
- **Core or extra.** A dependency the registry needs at import is a core dependency; one only
  some tasks use, and import inside a function, is an extra.
- **Updating** any pinned piece is a deliberate change: bump the pin (for vendored code,
  re-copy and re-apply the `NOTICE.md` changes), check the results against upstream's own on
  the benchmark's data where possible, and say in the commit message that scores may move.

## Consequences

Upgrades happen only when someone makes them, so upstream fixes arrive late; installs may
conflict more often with other packages in a user's environment, which is why only what grades
is pinned.

## Alternatives considered

- Lower bounds everywhere: the easiest install, but a release of an evaluator moves scores
  silently.
- Vendoring every upstream piece: no dependency conflicts, but every update is a manual re-copy.
