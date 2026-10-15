# DNA-Eval

**Do Not Approximate. Evaluate.**

The DNA evaluation suite: the LLM-judged tasks used to evaluate the DNA model family, built on
[Inspect AI](https://inspect.aisi.org.uk). Each answer of the model under evaluation is scored
by a judge model against a rubric, over 15 runs, and reported as a trimmed mean of the runs.
How answers are generated, judged and aggregated is described in
[docs/methodology.md](docs/methodology.md).

## Tasks

| task | data | what it measures | rubrics |
| --- | --- | --- | --- |
| `performance` | `dnotitia/dna_performance` (30 prompts) | general Korean QA against reference answers | `performance` |
| `hallucination` | `dnotitia/dna_hallucination` (100 prompts, 5 categories) | false premises, nonexistent entities, unknowable facts, grounding | `hallucination` (also per category) |
| `persona` | `dnotitia/dna_persona` (50 questions) | whether the model describes itself and its maker correctly | `persona` |
| `uncensorship` | `dnotitia/dna_uncensorship` (50 prompts) | whether the model engages with topics other base models refuse | `uncensorship` (`uncensorship_bias` and `uncensorship_quality` combined) |
| `degeneration` | `dnotitia/dna_confusion` (45 prompts) | whether a right answer breaks down: language confusion, repetition | `language_confusion`, `repetition` |
| `style` | `dnotitia/dna_style` (50 prompts) | whether the model reasons and answers in the style of a reference model | `reasoning_style`, `content_style` |

Every task runs 15 times by default (`-T epochs=`), needs a judge bound to the `grader` role,
and records the suite version (`SUITE_VERSION` in `src/dna_eval/tasks/__init__.py`) in its log.
Scores are comparable only within one suite version.

## Quickstart

```bash
pip install -e .
export HF_TOKEN=...            # read access to the dnotitia/dna_* datasets

# model under evaluation: any OpenAI-compatible endpoint (vLLM provider shown)
export VLLM_BASE_URL=http://localhost:8000/v1  VLLM_API_KEY=EMPTY
# judge: a second OpenAI-compatible endpoint, bound to the "grader" role
export JUDGE_BASE_URL=http://localhost:8001/v1  JUDGE_API_KEY=EMPTY

inspect eval dna_eval/performance \
  --model vllm/<model> \
  --model-role grader=openai-api/judge/gemma-4-31B-it \
  -T thinking=true -T epochs=15
```

The summary shows `trimmed_mean` first, then `trimmed_passed` and `runs`, `mean`, and two
failure rates:

- `call_failures`: the share of answers where every call to the model failed (`answer_failed`)
  or returned nothing (`answer_empty`)
- `judge_failures`: the share where the judge still failed after its retries (`judge_failed`)

These samples stay in the averages (a placeholder answer is judged; a failed judgment scores 0),
so a high rate means a low score reflects the endpoint, not the model. `inspect view` shows
every sample, judge prompt and judge reason.

Requests to the model have no time limit by default. `--attempt-timeout <seconds>` caps each
request; a request that runs out is retried and, if every attempt fails, judged as an empty
answer.

The `style` task also needs a reference model (`-T reference=...`); see
[docs/running-style-task.md](docs/running-style-task.md).

## Environment variables

Variables can also go in a `.env` file, which Inspect loads from the working directory upward.

| variable | used for |
| --- | --- |
| `HF_TOKEN` | reading the `dnotitia/dna_*` datasets |
| `VLLM_BASE_URL`, `VLLM_API_KEY` | the server of the model under evaluation (`--model vllm/...`); unset, Inspect starts a local vLLM server |
| `JUDGE_BASE_URL`, `JUDGE_API_KEY` | the judge server, for `--model-role grader=openai-api/judge/...` (the service name in `openai-api/judge/...` picks the variables) |
| `DNA_EVAL_CACHE_DIR` | where generated style references are kept (default `~/.cache/dna_eval`) |

Defaults, `.env` precedence, model roles and the cache layout:
[docs/configuration.md](docs/configuration.md).

## Layout

```
src/dna_eval/
  tasks/             one @task per file; __init__.py holds the suite's conventions
                     (SUITE_VERSION, generation settings, headline metric)
  rubrics/           the YAML rubrics and their schema
  rubric_judging.py  the judge prompt, judge calls, verdict reading, re-judge rounds
  solvers/           dna_generate (calls the model), style_reference (style references)
  scorers/           rubric_scorer: one registered scorer per rubric
  metrics/           per_run_trimmed, call_failures, judge_failures
  dataset.py         loader for dnotitia/dna_* at a pinned revision
  roles.py           model role names (grader, reference)
  utils/             helpers (JSON, model output, retries, statistics, templates, cache)
tests/
  parity/            the suite reproduces DeepEval GEval: prompts, verdict reading, failures
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest                      # all tests
python -m pytest -m "not network"     # without Hub downloads
```

Tests never call a real model: they use `mockllm` and `tests/parity/fake_openai.py`. A change
that moves scores (the judge prompt, a rubric, generation settings, a dataset revision) raises
`SUITE_VERSION` and says so in [docs/methodology.md](docs/methodology.md). Fixtures under
`tests/fixtures/` are generated by their scripts, never edited by hand.

## License

Apache License 2.0 (`LICENSE`), copyright Dnotitia Inc. The judge prompt and the JSON reader
are ported from DeepEval (Apache-2.0); see `NOTICE`. The datasets are downloaded at run time
under their own terms.
