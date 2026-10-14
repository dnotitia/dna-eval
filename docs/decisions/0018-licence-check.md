# 0018. Licences are checked by one short procedure; a script reports, maintainers decide

- Status: accepted
- Date: 2026-10-14
- Scope: adding or changing a task, and any third-party text or code.

## Context

Tasks download data under many licences (some non-commercial, one with none stated) and reuse
prompts and code from benchmark authors. Contributors are people and AI agents; a check that
needs several documents, fixed comment formats or hand-kept tables in sync is easy for an
agent and a burden for a person.

## Decision

- One procedure, in one place: "Before you start: licences" in `docs/adding-a-task.md`.
- What can be looked up is looked up by `scripts/license_report.py`: it finds the data each
  task uses without any declaration and reports the licence and access terms at the pinned
  revision against the table in `docs/benchmarks.md`. It runs on every pull request and only
  reports.
- What only the contributor knows, whether text or code was copied, is asked once, in free form,
  in the pull request template. Copied material gets a source comment and a
  `THIRD_PARTY_NOTICES.md` entry; non-commercial text is marked as an SPDX snippet where it is used.
- Maintainers decide non-commercial, share-alike, unlicensed and unclear cases, and tidy the
  notices in review.

## Consequences

The check cannot find copying nobody declares; review has to. A blocking check can be added
later if the report proves reliable.

## Alternatives considered

- A blocking CI gate with required declarations: precise, but every contributor pays for it.
- An agent-only skill: helps one tool's users, not people or other agents.
