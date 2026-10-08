# Adding a task

A task is four parts glued together in one file under `src/dna_eval/tasks/`:

| part | what it is | where it comes from |
| --- | --- | --- |
| dataset | `Sample(id, input, target, metadata)` rows | any Inspect loader, or `dna_dataset` for suite datasets |
| solver | how the model under evaluation is called | `dna_generate()` plus Inspect solvers or `solvers/` |
| scorer | how one sample is scored | `rubric_scorer` + a YAML rubric, or `scorers/` |
| epochs + config | how many runs, and the generation settings | `epochs=`, `generation.dna_generate_config` |

The walkthrough adds a hypothetical task, **`my_task`**. Its data is a Hugging Face dataset
`your-org/my_dataset` with `question` and `answer` columns, and an LLM judge grades each answer
against the reference. That is the common case in this suite: the judge scores every answer
against a rubric, 15 runs are folded into a trimmed mean and a pass count, and the task file
stays a few lines long.

## 1. Dataset

```python
dataset=hf_dataset("your-org/my_dataset", split="train", revision=revision,
                   sample_fields=FieldSpec(input="question", target="answer")),
```

`FieldSpec` maps your columns onto Inspect's `Sample`: `question` becomes the model's input and
`answer` becomes the target the judge compares against. Columns listed in `metadata=[...]` are
carried along; a `context` column is shown to the judge if the rubric asks for it. A dataset
whose rows need more than a column rename goes through `record_to_sample` instead.

- **Pin the commit.** Expose `revision` as a task argument whose default is the commit your
  reported numbers used, so it lands in the log's `task_args`. `-T revision=main` opts into
  the latest data.
- **Suite datasets** (`dnotitia/dna_*`) share one schema (`id`, `input`, optional `target`) and
  load with `dna_dataset("dna_<name>", revision=...)`, which requires the revision.

## 2. Solver

```python
solver=dna_generate(),
```

For most tasks that is the whole chain: the sample's `input` becomes one user turn and the
model answers. `dna_generate` is Inspect's `generate()` plus the suite's handling of the
answer before it is judged:

- a call that errors, or returns neither content nor reasoning, is retried (3 attempts);
- the answer is stripped of surrounding whitespace;
- no answer at all is judged as `[모델의 응답이 비어 있음]` instead of the sample erroring out.

A task that does not want this handling can use plain `generate()`. The suite's other
conventions behind that single line:

- no system prompt unless the task is about one;
- sampling left to the served model's `generation_config`; never set temperature here;
- thinking is a task argument forwarded through `dna_generate_config(thinking)`.

If `my_task` needs a system prompt, prepend a solver:

```python
# src/dna_eval/solvers/my_system_prompt.py
@solver
def my_system_prompt(text: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.insert(0, ChatMessageSystem(content=text))
        return state
    return solve
```

```python
solver=[my_system_prompt("Answer in one sentence."), dna_generate()],
```

A solver goes in `solvers/` (one `@solver` per file) when another task may reuse it. A prompt
change that only `my_task` needs can stay inline (`[prompt_template(...), dna_generate()]`).

## 3. Scorer: write a rubric, not code

```python
scorer=rubric_scorer("my_task"),
```

`rubric_scorer` is one judge pass driven by `src/dna_eval/rubrics/my_task.yaml`:

```yaml
name: my_task
params: [input, actual_output, expected_output]
steps:
  - Identify the final answer in expected_output.
  - Check whether actual_output reaches the same answer.
  - Penalise claims in actual_output that contradict expected_output.
rubric:
  - {range: [0, 3],  outcome: Wrong answer or contradicts the reference.}
  - {range: [4, 6],  outcome: Partly correct; key parts missing.}
  - {range: [7, 10], outcome: Same answer as the reference.}
threshold: 0.7
```

| field | meaning |
| --- | --- |
| `params` | which fields the judge is shown: `input`, `actual_output`, `expected_output`, `context` |
| `steps` | the evaluation steps, verbatim; the judge reads them numbered |
| `rubric` | score bands on the integer scale and what each band means |
| `threshold` | pass line on the normalised 0-1 score (0.7 = judge gave 7 or more) |

The rubric is validated on load (`rubrics/__init__.py`, pydantic). The judge **prompt** that
wraps it, `prompts/geval_results.jinja`, is shared by every task in the suite and reproduces the
DeepEval GEval prompt byte for byte; you edit rubrics, not the prompt.

Variations:

- **Several judge passes on one answer** (as in `uncensorship`): write one YAML per pass and
  compose `composite_scorer([rubric_scorer("a"), rubric_scorer("b")], names=["a", "b"])`.
  The sample score is the mean, and a sample passes only if every pass clears its threshold.
- **Two independent metrics from one answer**: `scorer=[rubric_scorer("a"), rubric_scorer("b")]`.
- **Not a judge task at all**: write your own `@scorer` in `scorers/`, one per file. For
  example, a string-match check:

  ```python
  # src/dna_eval/scorers/my_scorer.py
  @scorer(metrics=[mean(), per_run_trimmed()])
  def my_scorer() -> Scorer:
      async def score(state: TaskState, target: Target) -> Score:
          passed = target.text.strip() in state.output.completion
          return Score(value=1.0 if passed else 0.0, answer=state.output.completion,
                       metadata={"passed": passed, "epoch": state.epoch})
      return score
  ```

  `per_run_trimmed` reads `epoch` and `passed` from `Score.metadata`, so a scorer that wants
  the suite's per-run statistics has to set both.

## 4. Runs, generation settings, metrics

```python
epochs=epochs,                         # default 15
config=dna_generate_config(thinking),  # max_tokens 16384, 3 retries, thinking via chat_template_kwargs
```

`epochs=15` runs every sample 15 times. Inspect would normally reduce those per sample; the
suite instead reports **per-run** statistics (the dataset mean of each run, 15% trimmed across
runs, plus the trimmed pass count) via the `per_run_trimmed` metric that `rubric_scorer`
already declares. Both numbers appear in the `inspect eval` summary and in `inspect view`.

`dna_generate_config(thinking, **overrides)` is the one place generation settings live. If
`my_task` overrides something (say, a larger `max_tokens`), say why in a comment: the override
moves its numbers off the suite convention.

To report something the scorer does not, add a metric (one `@metric` per file in `metrics/`)
and list the task's metrics explicitly. `scores="unreduced"` hands the metric every run of
every sample rather than one reduced score per sample:

```python
# src/dna_eval/metrics/my_metric.py
@metric(scores="unreduced")
def my_metric() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        by_run = defaultdict(list)
        for s in scores:
            by_run[s.score.metadata["epoch"]].append(s.score.as_float())
        return min(sum(v) / len(v) for v in by_run.values())   # lowest run mean
    return compute
```

```python
metrics=[mean(), per_run_trimmed(), my_metric()],   # replaces the scorer's own metrics
```

## 5. Assemble and register

```python
# src/dna_eval/tasks/my_task.py
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset

from dna_eval.generation import dna_generate_config
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate

DATASET_REVISION = "<commit sha>"


@task
def my_task(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    return Task(
        dataset=hf_dataset("your-org/my_dataset", split="train", revision=revision,
                           sample_fields=FieldSpec(input="question", target="answer")),
        solver=dna_generate(),
        scorer=rubric_scorer("my_task"),
        epochs=epochs,
        config=dna_generate_config(thinking),
    )
```

Task arguments are the knobs a run can turn (`-T thinking=false -T epochs=3`); keep them to
what genuinely varies between runs. Register the task in `src/dna_eval/_registry.py`:

```python
from dna_eval.tasks.my_task import my_task  # noqa: F401
```

## 6. Test

Install the test dependencies with `pip install -e ".[dev]"`, then run `pytest` from the
repository root.

- **Offline, no model:** run the task on `mockllm` to see the chain execute.

  ```python
  from inspect_ai import eval
  from inspect_ai.model import get_model, ModelOutput
  judge = get_model("mockllm/model", custom_outputs=(ModelOutput.from_content("mockllm/model", '{"score": 8, "reason": "ok"}') for _ in iter(int, 1)))
  eval(my_task(epochs=2), model="mockllm/model", model_roles={"grader": judge}, limit=3)
  ```

- **If the task reproduces a DeepEval GEval metric**, add it to `tests/parity/cases.py` with
  its two fixtures (the judge prompt rendered by DeepEval, the per-run means and pass counts of
  a DeepEval run). The parity tests parameterised over `cases.py` then cover it.

## 7. Run

```bash
export VLLM_BASE_URL=http://localhost:8000/v1 VLLM_API_KEY=EMPTY
export JUDGE_BASE_URL=http://localhost:8001/v1 JUDGE_API_KEY=EMPTY
inspect eval dna_eval/my_task --model vllm/<model> \
  --model-role grader=openai-api/judge/<judge-model> -T thinking=true
```
