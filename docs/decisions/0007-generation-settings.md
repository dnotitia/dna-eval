# 0007. Generation settings of the DNA suite

- Status: accepted
- Date: 2026-10-12
- Scope: tasks in `tasks/dna_suite/`. Other tasks set their own generation settings.

## Context

The reference numbers were produced with fixed request settings. Every setting that changes
what the model sees or how long it may answer moves scores.

## Decision

- One user turn, no system prompt; the persona task shows its persona sheet to the judge only.
- Sampling is left to the served model's `generation_config`; no temperature is sent.
- `max_tokens` is 16384, the reference harness's committed setting; thinking is on by default
  and toggled through `chat_template_kwargs`.
- These live in `tasks/dna_suite/__init__.py` (`dna_generate_config`). A task that overrides one
  says why in its own file.
- No per-request timeout by default. To compare with numbers from the reference harness, run
  with `--attempt-timeout 180` (its per-request limit).

## Consequences

Long reasoning can be cut at 16384 tokens. A larger budget is a suite version change, recorded
by superseding this record.
