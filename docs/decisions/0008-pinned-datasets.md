# 0008. Datasets are pinned to a Hub commit

- Status: accepted
- Date: 2026-10-12

## Context

The suite's datasets live on the Hugging Face Hub (`dnotitia/dna_*`). An update to a dataset
would otherwise move published numbers without any change in this repository.

## Decision

- `dataset.dna_dataset` requires a `revision`. Each task pins it as the default of its own
  `revision` argument, so the commit is recorded in the log's `task_args`.
- Files stored next to the data (the persona sheet, style references) are read at the same
  revision (`dna_file`, `dna_table`, `dna_files`).
- `-T revision=main` opts into the latest data explicitly.

## Consequences

Updating a dataset is a two-step change: push the data, then bump the pinned commit in the task.
