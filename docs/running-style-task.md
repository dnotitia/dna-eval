# Running the style task

`style` asks whether a model **reasons and answers in the style of a reference model**: the
same voice, structure, register, format, length and vocabulary, regardless of whether the
content is correct. It is usually run to check that fine-tuning kept the style of the base
model, so the reference is normally the same-size base model the evaluated model came from.

The task has more moving parts than the other suite tasks: a second model (the reference),
references that are either published or generated, and two judges that look at different
parts of each answer. This guide walks through them.

## What happens in one run

For each of the 50 prompts in `dnotitia/dna_style`, and for each of the `epochs` runs (15 by
default):

1. **Reference.** The reference model's reasoning and answer for the prompt are attached to the
   sample, from published references or from the local cache (step-by-step below).
2. **Answer.** The model under evaluation answers once (`dna_generate`). Its reasoning (from a
   `reasoning_content` / `reasoning` field or a `<think>` block) is split from the final answer.
3. **Judging.** Two judges score the same answer independently, each against the reference:

   | scorer | compares | rubric |
   | --- | --- | --- |
   | `reasoning_style` | the model's reasoning with the reference's reasoning | `rubrics/reasoning_style.yaml` |
   | `content_style` | the model's answer with the reference's answer | `rubrics/content_style.yaml` |

   The judge is not shown the prompt, only the two texts: the rubrics say to ignore content
   and correctness. Each scorer reports its own `trimmed_mean` and `trimmed_passed` over runs;
   they are not combined.

The reference is the same in every run: it is produced once per prompt, and only the evaluated
model's answers vary between runs.

## Choosing a reference

`reference` is required and has no default, because a style score only means something against
the right model. Use the base model of the model under evaluation, at the same size.

References published with the dataset (at the revision the task is pinned to):

```
Qwen3.5-0.8B   Qwen3.5-2B   Qwen3.5-4B   Qwen3.5-9B   Qwen3.6-27B   Qwen3.6-35B-A3B   Qwen3.5-122B-A10B
```

The 0.8B and 2B references were generated with thinking off, so they have no reasoning; the
`reasoning_style` judge then sees `[참조 모델 추론 없음]` as the reference and that score is not
informative for those sizes.

## Case 1: a published reference

```bash
export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY
export JUDGE_BASE_URL=http://localhost:8001/v1 JUDGE_API_KEY=EMPTY

inspect eval dna_eval/style \
  --model vllm/<model under evaluation> \
  --model-role grader=openai-api/judge/<judge model> \
  -T reference=Qwen3.6-27B
```

Nothing else is needed: the references are read from the dataset repository at the pinned
revision, so every run of the suite compares against exactly the same texts. This is the
setting to use when a number has to be comparable with published ones.

## Case 2: a model without published references

Bind the reference model to the `reference` role. Its responses are generated on the first run
and cached locally; later runs reuse the cache.

```bash
inspect eval dna_eval/style \
  --model vllm/<model under evaluation> \
  --model-role grader=openai-api/judge/<judge model> \
  --model-role reference=openai-api/ref/<reference model> \
  -T reference=<reference model>
```

`openai-api/ref/...` reads `REF_BASE_URL` / `REF_API_KEY`, like the judge's `JUDGE_*`. Any Inspect
model spec works for the role.

- **Generated once per prompt.** Runs execute in parallel, so the first run to reach a prompt
  generates its reference while the others wait for it. A 15-run evaluation therefore makes
  50 reference calls, not 750.
- **Same handling as the evaluated model.** Reference calls use the task's generation settings
  (thinking, `max_tokens`) and the same retry rule and answer/reasoning split as `dna_generate`.
- **Failures are not cached.** If every attempt fails, the judge sees `[참조 모델 응답 없음]` for
  that prompt in this evaluation, and the next evaluation tries again.
- **The role is required.** Without `--model-role reference=...` the evaluation stops with
  `Model role 'reference' is required`; it never falls back to the evaluated model.

`-T reference_source` chooses explicitly:

| value | references used |
| --- | --- |
| `auto` (default) | published ones if `reference` has them, otherwise generated with the role |
| `hub` | published ones only; an unknown `reference` is an error listing the published names |
| `generate` | always the role and the local cache, even if published references exist |

With `auto`, a published reference wins: if the `reference` role is also bound, the run logs a
warning that the role is ignored. Use `reference_source=generate` to compare against a fresh
generation from a model that also has published references.

## The local cache

```
$DNA_EVAL_CACHE_DIR (default ~/.cache/dna_eval)
  style/<dataset revision>/<reference>.<settings hash>.parquet
  style/<dataset revision>/<reference>.<settings hash>.json
```

- The **parquet** file has the published references' schema: `id`, `input`,
  `reference_reasoning`, `reference_content`.
- The **json** file records where it came from: the reference name, the model spec the role was
  bound to, the generation settings, the dataset revision, the row count and the last update.
- The **settings hash** covers the role's model and the generation settings, so a different
  model or e.g. `-T thinking=false` gets its own file instead of reusing incompatible
  references. Connection and retry settings are left out of it.

The cache is keyed by model *name*: if different weights are served under the same name, the
cache cannot tell. Regenerate with

```bash
inspect eval dna_eval/style ... -T refresh_reference=true
```

which ignores the existing file and overwrites it.

### Publishing a cached reference

Because the file already has the published schema, maintainers can add it to the dataset as it
is, then pin the task to the new revision:

```bash
huggingface-cli upload dnotitia/dna_style \
  ~/.cache/dna_eval/style/<revision>/<reference>.<hash>.parquet references/<reference>.parquet \
  --repo-type dataset
# then set DATASET_REVISION in src/dna_eval/tasks/style.py to the new commit
```

Check the `.json` provenance first: the reference should have been generated with the
settings the suite uses (thinking on, the suite's `max_tokens`).

## Running one judge

```bash
inspect eval dna_eval/style ... -T reference=Qwen3.6-27B -T judges=content_style
```

## Reading the results

In `inspect view`:

- The log header shows `reasoning_style` and `content_style` with their per-run statistics.
- Each sample's **Scoring** tab has both judgments with the judge's reason.
- The sample **Metadata** holds `reference_reasoning` / `reference_content`, and the sample store
  has `reasoning` (the evaluated model's reasoning as the judge saw it) and `reference_origin`
  (`hub`, `cache` or `generated`).
- The task metadata records `reference` and `reference_source` (`hub` or `local`).

## Comparing with DeepEval-based results

With published references the task reproduces the DeepEval GEval setup it was ported from: the
judge prompts are byte-identical, the placeholders for missing reasoning or answers are the
same strings, and the reasoning is split from the answer the same way (covered by
`tests/parity/`). Generated references are equivalent in procedure (one response per prompt,
same generation settings) but are new samples from the reference model, so scores against them
are not directly comparable with scores against published references.
