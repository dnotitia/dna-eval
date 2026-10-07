# dna-eval

Evaluation suite for the DNA model family, built on [Inspect AI](https://inspect.aisi.org.uk).
A task is four replaceable parts — **dataset** (`dnotitia/dna_*` on the Hugging Face Hub),
**generation** (Inspect solvers), **scoring** (one rubric-driven LLM judge), and
**aggregation** (`dna_eval.metrics`). Rubrics are data (`src/dna_eval/rubrics/*.yaml`), not code.

> Status: `dev` branch, quickstart scope. Only the `performance` task is wired up.
> The datasets are private for now; access is required to run anything.

## Quickstart

```bash
pip install -e .
export HF_TOKEN=...            # read access to dnotitia/dna_* datasets

# subject: any OpenAI-compatible endpoint (vLLM provider shown)
export VLLM_BASE_URL=http://localhost:49002/v1  VLLM_API_KEY=EMPTY
# judge: a second OpenAI-compatible endpoint, bound to the "grader" role via the generic provider
export JUDGE_BASE_URL=http://localhost:8056/v1  JUDGE_API_KEY=EMPTY

inspect eval dna_eval/performance \
  --model vllm/Qwen3.8-Flash-Next \
  --model-role grader=openai-api/judge/gemma-4-31B-it \
  -T thinking=true -T epochs=15

```

The summary reports `mean` plus the suite's per-run statistics `trimmed_mean`, `trimmed_passed`, `runs`.
`inspect view` shows every sample, judge prompt, and judge reason.

## How a task is put together

```python
@task
def performance(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    return Task(
        dataset=dna_dataset("dna_performance", revision=revision),   # id / input / target
        solver=generate(),
        scorer=rubric_scorer("performance"),                          # scorers/ + rubrics/performance.yaml
        epochs=epochs,
        config=GenerateConfig(extra_body={"chat_template_kwargs": {"enable_thinking": thinking}}),
    )
```

- **Rubric** (`rubrics/<name>.yaml`): evaluation steps, score bands, which fields the judge
  sees, pass threshold. Validated by the pydantic model in `rubrics/__init__.py`.
- **Judge prompt** (`prompts/geval_results.jinja`): fixed for the whole suite. It reproduces
  the DeepEval GEval template byte-for-byte so scores remain comparable with earlier
  DeepEval-based measurements. Changing it is a scorer version bump, not a rubric edit.
- **Judge**: whatever model is bound to the `grader` role; called at temperature 0, expected
  to answer `{"score": 0-10, "reason": ...}`. Two retries on unparsable output, then 0.

## Per-run statistics (epochs fold the other way)

Inspect's epoch reducers fold the N runs of *one sample*. The suite convention folds *per
run*: average the dataset within each run, then take a 15% trimmed mean across runs, plus
the trimmed mean of per-run pass counts. Both agree on the plain mean; they differ on the
trimmed mean and the pass count. `rubric_scorer` stamps `state.epoch` and `passed` into every
score, and the `per_run_trimmed` metric (`@metric(scores="unreduced")`) regroups by run.

## Layout

```
src/dna_eval/
  datasets.py        shortcut loader for dnotitia/dna_* (revision required)
  generation.py      the suite's GenerateConfig (max_tokens, retries, thinking toggle)
  rubrics/           rubric schema + the YAML rubrics
  prompts/           judge-prompt rendering + Jinja templates
  solvers/           one @solver per file
  scorers/           one @scorer per file
  metrics/           one @metric per file
  utils/             small dependency-free helpers
  tasks/             one @task per file
```

## Adding a task (planned shape)

1. Prepare the dataset. Any Inspect `Dataset` works: `rubric_scorer` only reads the sample's
   input, target and `metadata["context"]`. For a `dnotitia/dna_*` dataset with columns `id`,
   `input`, optional `target`, use `dna_dataset`; otherwise use `hf_dataset` / `json_dataset`
   with a `FieldSpec` or `record_to_sample` in the task file.
2. Add `rubrics/<name>.yaml`.
3. Add `tasks/<name>.py` — a `@task` that picks the dataset, solver(s) and `rubric_scorer("<name>")`.
   Pin the dataset commit as the default of a `revision` argument (full SHA), so the log's
   `task_args` records it; `-T revision=main` runs against the latest data.

## Tests

`pytest tests/` — unit tests plus the DeepEval parity suite (`tests/parity/`, see
`docs/parity-tests.md`). Every task must keep T1-T3 green; adding a task means adding its
fixtures and one entry in `tests/parity/cases.py`.

## License

MIT.
