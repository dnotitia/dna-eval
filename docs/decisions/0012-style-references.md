# 0012. Style references: published first, otherwise generated and cached locally

- Status: accepted
- Date: 2026-10-12

## Context

The style task compares answers with a reference model's answers. Published references (in the
dataset repository) make numbers reproducible; models without them must still be evaluable.

## Decision

- `reference` is a required task argument with no default.
- `reference_source=auto` uses published references when they exist, otherwise the model bound
  to the `reference` role; `hub` and `generate` force one side.
- Generated references are made once per sample (a per-sample lock across concurrent runs), with
  the suite's generation settings and `dna_generate`'s handling; failed calls are not cached.
- The cache is parquet in the published schema (`id`, `input`, `reference_reasoning`,
  `reference_content`) under `$DNA_EVAL_CACHE_DIR/style/<revision>/`, keyed by a hash of the
  role's model and settings, with a `.json` provenance file. It can be uploaded as it is.

## Consequences

Scores against generated references are not comparable with scores against published ones;
the log's task metadata says which was used. Same-name weight changes need `refresh_reference`.

## Alternatives considered

- Inspect's model cache: pickled, no per-sample lock (concurrent runs generate twice), not uploadable.
