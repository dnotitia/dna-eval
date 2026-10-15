# Methodology

How the suite turns a model's answers into its numbers. Everything here can move a score: a
change to any of it raises `SUITE_VERSION` (`src/dna_eval/tasks/__init__.py`) and is listed under
[Suite versions](#suite-versions).

The judging reproduces DeepEval GEval: the judge prompt, the reading of the judge's reply and
the handling of failed calls match it, and the tests under `tests/parity/` check that they do.

## 1. Data

- Each task reads a `dnotitia/dna_*` dataset on the Hugging Face Hub at a pinned commit. The
  commit is the default of the task's `revision` argument and is recorded in the log.
- Files stored with a dataset (the persona sheet, published style references) are read at the
  same commit.
- `-T revision=main` reads the latest data instead; such a run is not comparable with pinned ones.

## 2. Generating answers

- One user turn with the sample's input; no system prompt. The persona task shows its persona
  sheet to the judge only.
- Sampling is left to the served model's `generation_config`; no temperature is sent.
- Answers may use up to 16,384 tokens, reasoning included; a longer answer is cut there.
- Thinking is on by default and switched with `-T thinking=false`, sent as
  `chat_template_kwargs.enable_thinking`.
- A call that fails, or returns neither an answer nor reasoning, is tried 3 times (1 s, then 2 s
  apart). An answer still missing is judged as `[모델의 응답이 비어 있음]`; missing reasoning is
  `[추론 없음]`. These placeholders are part of the behaviour.
- Requests have no time limit unless `--attempt-timeout` is given.

## 3. Judging

- The judge is the model bound to the `grader` role; a run without one stops at the first sample.
- A rubric (`src/dna_eval/rubrics/<name>.yaml`) gives the evaluation steps, the score bands and
  the pass threshold, and names what the judge sees (`params`: input, answer, reasoning,
  reference answer, context).
- Every rubric is rendered into the same prompt, DeepEval's GEval results prompt, byte for byte
  (`GEVAL_RESULTS` in `src/dna_eval/rubric_judging.py`). The judge returns a score within the
  rubric's range and a reason; the sample's score is that score scaled to 0-1.
- The reply is read as GEval reads it: the text from the first `{` to the last `}`, a missing
  closing brace added, trailing commas dropped, then `score` and `reason` validated. Scores are
  not clipped. A reply that still fails is read once more with stray backslashes escaped.
- A failed or unreadable judgment is retried for 2 more rounds and then scores 0, still counted.
- A composite rubric (`uncensorship`) judges the same answer with each of its parts: the score is
  their mean, and the sample passes only if every part passes. When one part fails, all parts
  are judged again.

## 4. Aggregating

- A task runs every sample `epochs` times (15 by default). Each run is a pass over the whole
  dataset.
- `trimmed_mean`, the headline: the mean score of each run, then the 15% trimmed mean of those
  run means (`int(runs × 0.15)` runs dropped from each end). Run means are not rounded.
- `trimmed_passed`: the same over each run's count of passed samples, where passed means the
  rubric's threshold was met.
- `hallucination` also reports both per category.
- `mean` (over all samples and runs) is reported as a cross-check.
- `call_failures` and `judge_failures` give the share of scores that stand for failed calls
  rather than judged answers.

## 5. Style references

- `style` compares each answer with a reference model's answer to the same prompt.
- `reference_source=auto` uses the references published with the dataset when they exist for
  `-T reference=`, otherwise generates them with the model bound to the `reference` role, with
  the same generation settings, and caches them locally.
- Scores against generated references are not comparable with scores against published ones;
  the log records which were used. See [running-style-task.md](running-style-task.md).

## Suite versions

| version | change |
| --- | --- |
| 1 | the suite as first released |
