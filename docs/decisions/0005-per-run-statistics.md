# 0005. Report per-run statistics: 15% trimmed mean of run means, and pass counts

- Status: accepted
- Date: 2026-10-12
- Scope: the DNA suite (0001). `per_run_trimmed` is available to any task; other tasks choose the
  metrics that fit them.

## Context

The suite runs every sample several times. Inspect's epoch reducers fold the runs of *one
sample*; the suite has always folded the other way: average each run over the dataset, then
summarise across runs, which damps judge and sampling noise.

## Decision

- `epochs` are runs (15 by default). `metrics/per_run_trimmed.py` regroups unreduced scores by
  the `epoch` the scorer stamps and reports `trimmed_mean` (15% trimmed, `int(n * p)` dropped
  from each end) and `trimmed_passed`, optionally per metadata group (`group_by`).
- The scorer's `metadata["passed"]` (the rubric threshold) is the only definition of "passed".
- Suite tasks set `headline_metric=HEADLINE` so the trimmed mean is shown first; otherwise
  Inspect shows `mean`, which it computes before unreduced metrics.
- Per-run means are not rounded. The reference harness rounded them to two decimals in its log;
  we do not reproduce that.

## Consequences

Numbers can differ from the old log-derived ones by the rounding (well under 0.005 per run).
`mean` is still reported, as a cross-check.

## Alternatives considered

- Inspect `grouped()` for categories: only wraps metrics that return a single number.
