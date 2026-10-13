# 0016. Standard benchmarks follow their official protocol, adapted only for reasoning models

- Status: accepted
- Date: 2026-10-12
- Scope: tasks in `tasks/benchmarks/` (GPQA, MMLU-Pro, HLE, AIME, IFEval, ...), not the DNA suite.

## Context

These tasks exist so that a model's numbers can be set next to those published by other
model reports. Benchmark authors publish evaluation code, but much of it predates reasoning
models: short output caps, the first regex match, greedy decoding, log-probabilities.

## Decision

- **Official protocol first.** Prompt, answer extraction, correctness check and metrics follow
  the benchmark authors' released code; where there is none, the most used reference (openai
  simple-evals, MathArena, lm-eval-harness), named in the task's docstring.
- **Adapt only for reasoning models, and record it.** Thinking on; sampling from the served
  model's `generation_config`; no `max_tokens`; answers read from the final content (never the
  reasoning) and from the *last* match. Every other departure is listed in the task docstring.
- **Failures count, and are reported.** A truncated answer or one without the required format
  scores 0; `answer_diagnostics` reports the truncated and format-failure rates and the accuracy
  without truncated samples.
- **Repeats for small sets.** `epochs` with the mean reducer gives pass@1 averaged over runs:
  8 for GPQA-Diamond, AIME, IMO-AnswerBench and IFBench; 1 elsewhere. Random choices (option
  shuffles) are seeded per sample and run.
- **Judges only where the protocol uses one** (HLE, IMO-AnswerBench), bound to the `grader`
  role (0010) and named in the log; a judge other than the official one makes the number
  comparable only to runs with the same judge.
- **Pinned data** (0008): a Hub revision, or a commit and checksum for files fetched elsewhere.

## Consequences

Numbers differ from runs of the original scripts where those scripts cap output or take the
first match; the docstrings say how. Implementations are this project's own; `inspect_evals`
tasks were not reused because their reasoning-model behaviour (output caps, last-line parsing,
truncation scoring) and fixed dataset revisions do not fit these rules.

## Alternatives considered

- Reusing `inspect_evals` tasks as they are: see Consequences.
- lm-eval-harness log-likelihood variants: need log-probabilities; not how reasoning models are reported.
