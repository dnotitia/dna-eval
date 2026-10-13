# 0014. Describe behaviour, not where it came from

- Status: accepted
- Date: 2026-10-12

## Context

dna-eval is an open-source project. Comments, docs, names and fixtures written during migration
referred to internal repositories, internal model names, log files and session-only labels,
which mean nothing to outside contributors.

## Decision

- Code, docs, test names and fixtures describe what a thing does and why, in terms a reader of
  this repository can check. They do not refer to internal projects, internal model or server
  names, internal file paths, or labels that exist only in someone's working notes.
- The external reference is named by its public identity: "DeepEval GEval".
- Documentation is in English.

## Consequences

Reasons that depended on internal context are restated as the behaviour they produce (e.g. "a
fixed token budget" rather than "matches tool X").
