# 0003. Rubrics are data

- Status: accepted
- Date: 2026-10-12

## Context

Judge criteria (evaluation steps, score bands, threshold) change more often than code and are
written by people who should not need to edit Python. The steps are long Korean sentences.

## Decision

- A rubric is a YAML file in `src/dna_eval/rubrics/`, validated on load by pydantic models in
  `rubrics/__init__.py`.
- A judge rubric declares `params` (what the judge sees) and, when a field must come from
  somewhere other than the default, `sources` (`answer`, `reasoning`, `input`, `target`,
  `metadata.<key>`, `store.<key>`).
- A composite rubric (`compose`, `combine: mean`, `pass: all`) folds several judge rubrics
  applied to the same answer into one score.
- Adding a judge metric means adding a YAML file, not code.

## Consequences

Changing a rubric's steps or bands changes scores: treat it as a suite version change. The
schema grows only when a rubric needs it (`combine` and `pass` accept one value each today).

## Alternatives considered

- JSON: no comments, long multi-line strings are hard to read and review.
- Rubrics as Python objects: needs code review for wording changes; harder for non-programmers.
- A dedicated composite scorer: replaced by composite rubrics so that a combined metric is data too.
