# 0013. Tests run without models, network endpoints or internal code

- Status: accepted
- Date: 2026-10-12

## Context

Tests must run on any contributor's machine and in CI. The parity tests of the ported suite
(0001) must also not depend on the private harness they compare against.

## Decision

- Model calls go to `mockllm` or to `tests/parity/fake_openai.py`, a scripted OpenAI-compatible
  server, bound by passing `base_url` to `get_model` (environment variables are not used: Inspect
  re-reads `.env`, which overrides them inside VS Code).
- Parity expected values come from DeepEval itself (`tests/fixtures/make_judge_outputs.py`, public
  `deepeval==4.0.3`) or are recorded runs; fixtures are regenerated, not hand-edited.
- Tests that need the Hub are marked `network`.

## Consequences

The suite (`python -m pytest`) runs in about a minute with no GPU. A behaviour that can only be
checked against a real model belongs in a separate, opt-in run, not in these tests.
