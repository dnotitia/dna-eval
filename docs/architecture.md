# Architecture

How dna-eval is put together. For *why* a part is the way it is, follow the record numbers
(`docs/decisions/`); for how to add a task, see `docs/adding-a-task.md`.

## One sample, end to end

```
dataset (Hub, pinned)      solvers                      scorers                       metrics
  Sample(input, target,  ─▶ style_reference (style)  ─▶ rubric_scorer("<rubric>")  ─▶ per_run_trimmed
         metadata)          dna_generate               └─ rubric_judging.judge_rounds  mean
                              writes store["reasoning"]    ├─ build_prompt              (per run, then
                                                           ├─ grader role (judge)        across runs)
                                                           └─ read_verdict
```

1. **Dataset.** `dataset.dna_dataset` loads `dnotitia/<name>` at a pinned commit (0008).
2. **Solvers.** `dna_generate` calls the model under evaluation with the suite's handling of
   empty and failed replies, strips the answer, and stores the reasoning (0006).
3. **Scorers.** `rubric_scorer` resolves the rubric's fields (`sources`), renders the judge
   prompt, calls the judge, reads the verdict, retries failed rounds (0003, 0004, 0006, 0015).
4. **Metrics.** `per_run_trimmed` regroups the scores by run and reports the trimmed mean and
   pass count; suite tasks make it the headline (0005). `call_failures` and `judge_failures`
   report how many scores stand for failed calls rather than judged answers (0006).

A task runs this for every sample and every run (`epochs`).

A folder under `tasks/` groups tasks that share conventions, which its `__init__.py` and
decision records define: `tasks/dna_suite/` (0001, 0005-0007) and `tasks/benchmarks/` (0016). A
new task goes in the folder whose conventions it follows; a task that follows neither gets its
own folder.

Standard benchmarks (`tasks/benchmarks/`, 0016) use the same four stages with other parts: the
dataset is a public one pinned by Hub revision or checksum; the solver is Inspect's `generate()`
(GPQA shuffles options first with `shuffled_choices`); the scorer reads the answer from the final
content (`letter_choice`, `math_answer`, `instruction_following`) or asks the judge (`hle_judge`,
`imo_answer_grader`); metrics are the benchmark's own plus `answer_diagnostics`.

## Modules

| module | holds | may import |
| --- | --- | --- |
| `tasks/dna_suite/` | the suite's tasks; `__init__.py`: its conventions (generation config, headline) | everything below |
| `tasks/benchmarks/` | standard benchmarks; `__init__.py`: their generation config (0016) | everything below |
| `solvers/` | one solver per file, with its own defaults and placeholders | `roles`, `utils`, other solvers |
| `scorers/` | `rubric_scorer`: one registered scorer per rubric file (0009); the benchmark scorers, each with its judge prompt | `rubric_judging`, `roles`, `rubrics`, `metrics`, `vendor`, `utils` |
| `metrics/` | one metric per file | `utils`; the key names other metrics define |
| `rubric_judging.py` | the rubric judge: its prompt (`GEVAL_RESULTS`), judge call, verdict reading, re-judge rounds, `sources` resolution | `roles`, `rubrics`, `utils` |
| `roles.py` | the model role names (`grader`, `reference`) (0010) | — |
| `rubrics/` | rubric schema and the rubric YAML files | — |
| `dataset.py` | Hub loaders for the suite's datasets and the files next to them | — |
| `utils/` | helpers with no suite logic: JSON, statistics, model output, retries, math answers, cache | — |
| `vendor/` | third-party checking code, unchanged, with its licence and notice (0017) | — |
| `_registry.py` | the package entry point; imports every module that registers objects | everything |

Dependencies point down the table; solvers and scorers never import each other or `tasks` (0011).

Prompts live where they are used (0019): a task's instructions to the model under evaluation in
the task file, a judge's prompt in its scorer (or, for rubric judging, in `rubric_judging.py`).
A prompt several modules share lives in the module they all depend on, never in one of them;
templates that need Jinja are rendered with `utils/templates.py`.

## Contracts between parts

Parts exchange data through named keys, never through imports:

| key | written by | read by |
| --- | --- | --- |
| `Score.metadata["epoch"]`, `["passed"]` | `rubric_scorer` | `per_run_trimmed` |
| `Score.metadata["raw_score"]`, `["components"]` | `rubric_scorer` | people, in the viewer |
| `state.store["reasoning"]` | `dna_generate` | rubrics with `sources: {...: store.reasoning}` |
| `state.store["no_answer"]` → `Score.metadata["no_answer"]` | `dna_generate`, copied by `rubric_scorer` | `call_failures` |
| `Score.metadata["judge_error"]` | `rubric_scorer`, `hle_judge`, `imo_answer_grader` | `judge_failures`, people in the viewer |
| `state.store["reference_origin"]` | `style_reference` | people, in the viewer |
| `metadata["context"]` | dataset, or `rubric_scorer(context=...)` | rubrics with `context` in `params` |
| `metadata["reference_reasoning"]`, `["reference_content"]` | style task, `style_reference` | `style` rubrics |
| `metadata["category"]` | dataset | `per_run_trimmed(group_by="category")`, grouped benchmark metrics |
| `Score.metadata["truncated"]`, `["extracted"]` | benchmark scorers | `answer_diagnostics` |
| `metadata["correct_letter"]` | `shuffled_choices` | `letter_choice(target_key=...)` |
| `Score.metadata["confidence"]` | `hle_judge` | `calibration_error` |
| `Score.metadata["strict"]`, `["loose"]` | `instruction_following` | `instruction_accuracy` |

## Registration

Inspect finds dna-eval through the `inspect_ai` entry point (`dna_eval._registry`). Tasks
resolve as `dna_eval/<task>`; every rubric resolves as a scorer `dna_eval/<rubric>`, which is
how `inspect score --scorer` re-scores a log (0009). Model roles: `grader` (the judge, every
task) and `reference` (style, when references are generated); both are required when used (0010).

## Vocabulary

- **run**: one pass over the dataset; Inspect calls it an epoch. Statistics are per run first.
- **suite**: the DNA tasks in `tasks/dna_suite/` and the conventions they share.
- **rubric**: a YAML file the judge applies. A *judge rubric* is one judge pass; a *composite
  rubric* folds several judge rubrics into one score.
- **sources**: where a judge-visible field comes from (`answer`, `reasoning`, `target`, ...).
- **judge**: the model bound to the `grader` role.
- **reference**: in the style task, the model whose style answers are compared with.
- **parity**: matching DeepEval GEval behaviour; required only for the DNA suite tasks ported
  from a DeepEval harness (0001), not for other tasks.
- **placeholder**: the fixed text the judge sees when a model gave no answer or no reasoning.
- **headline**: the metric Inspect shows first for a task (`trimmed_mean` for the suite; the
  benchmark's own number for a standard benchmark).
- **benchmark**: a published evaluation reproduced in `tasks/benchmarks/` (0016).
- **truncated**: the model stopped at a token or context limit; such an answer scores 0.
