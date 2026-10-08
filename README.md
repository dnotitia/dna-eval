# DNA Evaluation Harness

An evaluation framework for the DNA model family, built on [Inspect AI](https://inspect.aisi.org.uk).

## Quickstart

```bash
pip install -e .
export HF_TOKEN=...            # read access to dnotitia/dna_* datasets

# subject: any OpenAI-compatible endpoint (vLLM provider shown)
export VLLM_BASE_URL=http://localhost:8000/v1  VLLM_API_KEY=EMPTY
# judge: a second OpenAI-compatible endpoint, bound to the "grader" role via the generic provider
export JUDGE_BASE_URL=http://localhost:8001/v1  JUDGE_API_KEY=EMPTY

inspect eval dna_eval/performance \
  --model vllm/<model> \
  --model-role grader=openai-api/judge/gemma-4-31B-it \
  -T thinking=true -T epochs=15

```

The summary reports `mean` plus the suite's per-run statistics `trimmed_mean`, `trimmed_passed`, `runs`.
`inspect view` shows every sample, judge prompt, and judge reason.

Requests to the model have no time limit by default. `--attempt-timeout <seconds>` caps each
request; a request that runs out is retried and, if every attempt fails, judged as an empty
answer. (`--timeout` is different: it bounds a call including all its retries.)


## Layout

```
src/dna_eval/
  dataset.py         shortcut loader for dnotitia/dna_* (revision required)
  generation.py      the suite's GenerateConfig (max_tokens, retries, thinking toggle)
  rubrics/           rubric schema + the YAML rubrics
  prompts/           judge-prompt rendering + Jinja templates
  solvers/           one @solver per file
  scorers/           one @scorer per file
  metrics/           one @metric per file
  utils/             small dependency-free helpers
  tasks/             one @task per file
```

## How to customize tasks (Simple)

To integrate your own dataset into a `dna-eval` task, add a new task file to `src/dna_eval/tasks`.

And then, add the followings if necessary: (1) dataset, (2) solver chain, (3) scorer, (4) run settings 

Each building block of dna-eval is based on Inspect AI; for anything not covered here, please refer to their documentations: [Tasks](https://inspect.aisi.org.uk/tasks.html) ·
[Datasets](https://inspect.aisi.org.uk/datasets.html) ·
[Solvers](https://inspect.aisi.org.uk/solvers.html) ·
[Scorers](https://inspect.aisi.org.uk/scorers.html) ·
[Metrics](https://inspect.aisi.org.uk/metrics.html) ·
[Model-graded scoring](https://inspect.aisi.org.uk/model-graded.html) ·
[Examples](https://inspect.aisi.org.uk/examples/).

Below is a simple example of adding a new task (called `my_task`) to the dna-eval harness: an LLM-judge task
whose criteria are stored as a YAML rubrics. Full walkthrough: `docs/adding-a-task.md`.

### Add a task

Say `your-org/my_dataset` on the Hugging Face Hub has `question` and `answer` columns, and an
LLM judge should grade each answer against the reference.

```python
# src/dna_eval/tasks/my_task.py
from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset

from dna_eval.generation import dna_generate_config
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate

DATASET_REVISION = "<commit sha>"   # pin the data so a dataset update cannot move your numbers

@task
def my_task(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    return Task(
        dataset=hf_dataset("your-org/my_dataset", split="train", revision=revision,
                           sample_fields=FieldSpec(input="question", target="answer")),
        solver=dna_generate(),                 # one user turn, no system prompt
        scorer=rubric_scorer("my_task"),       # rubrics/my_task.yaml
        epochs=epochs,                         # 15 runs -> per_run_trimmed
        config=dna_generate_config(thinking),  # generation settings
    )
```

A suite dataset (`dnotitia/dna_*`) loads with `dna_dataset("dna_<name>", revision=...)` instead.
The judge criteria go in one YAML file next to the other rubrics:

```yaml
# src/dna_eval/rubrics/my_task.yaml
name: my_task
params: [input, actual_output, expected_output]   # what the judge is shown
steps:
  - Identify the final answer in expected_output.
  - Check whether actual_output reaches the same answer.
  - Penalise claims in actual_output that contradict expected_output.
rubric:
  - {range: [0, 3],  outcome: Wrong answer or contradicts the reference.}
  - {range: [4, 6],  outcome: Partly correct; key parts missing.}
  - {range: [7, 10], outcome: Same answer as the reference.}
threshold: 0.7                                     # pass line on the 0-1 score
```

Register the task in `src/dna_eval/_registry.py`, and run it like any other task:

```python
from dna_eval.tasks.my_task import my_task  # noqa: F401
```

```bash
inspect eval dna_eval/my_task --model vllm/<model> --model-role grader=openai-api/judge/<judge-model>
```

The three sections below are needed only when `dna_generate()`, `rubric_scorer`, or the default
metrics are not enough. Each adds one file and changes one line of `my_task.py`.

### Add a solver

Say `my_task` should run with a system prompt. One `@solver` per file in `solvers/`:

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
# my_task.py
solver=[my_system_prompt("Answer in one sentence."), dna_generate()],
```

A prompt change that only `my_task` needs can stay inline in the task instead (e.g.
`prompt_template(...)` before `dna_generate()`).

### Add a scorer

Say `my_task` should be checked by string match instead of a judge. One `@scorer` per file in
`scorers/`:

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

```python
# my_task.py
scorer=my_scorer(),
```

`per_run_trimmed` reads `epoch` and `passed` from `Score.metadata`; a scorer that wants the
suite's per-run statistics has to set both.

### Add a metric

Say `my_task` should also report its worst run. One `@metric` per file in `metrics/`;
`scores="unreduced"` gives the metric every run of every sample instead of one reduced score
per sample:

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
# my_task.py (replaces the metrics the scorer declares)
metrics=[mean(), per_run_trimmed(), my_metric()],
```


## License

MIT.
