# 0006. Failed and empty calls are scored, as in the reference harness

- Status: accepted
- Date: 2026-10-12
- Scope: the ported DNA suite (0001). `dna_generate` is opt-in; other tasks may use plain
  `generate()` and Inspect's error handling. The judge retry applies wherever `rubric_scorer` is used.

## Context

By default Inspect marks a sample whose model call fails as an error, which either stops the
eval or drops the sample from the averages. The reference harness instead scored every sample,
which keeps the denominator fixed and penalises unreliable models.

## Decision

- Model under evaluation (`solvers/dna_generate.py`): a call that raises or returns neither
  content nor reasoning is retried (3 attempts, 1 s then 2 s apart); the answer is stripped;
  no answer is judged as `[모델의 응답이 비어 있음]`; missing reasoning is `[추론 없음]`.
- Judge (`rubric_judging.judge_rounds`): a failed or unreadable reply is retried for 2 more rounds and
  then scores 0, still counted. For a composite rubric, one failing component re-judges all of
  them, as DeepEval re-evaluates a whole test case.
- The placeholder strings are part of the behaviour and stay as they are.

## Consequences

Evals finish even when endpoints flake; flakiness shows up as low scores, not missing samples.
Retry counts differ from the reference in raw requests (provider-level retries) but not in outcome.

## Alternatives considered

- Inspect defaults (`fail_on_error`): different denominators, numbers not comparable.
