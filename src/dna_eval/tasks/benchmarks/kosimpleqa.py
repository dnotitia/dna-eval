"""KoSimpleQA: 1,000 short fact-seeking questions on Korean cultural knowledge, graded as SimpleQA."""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_ai.solver import generate

from dna_eval.metrics.answer_diagnostics import answer_diagnostics
from dna_eval.metrics.judge_failures import judge_failures
from dna_eval.scorers.simpleqa_grade import simpleqa_grade, simpleqa_metrics
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config

# The authors' release, github.com/naver-ai/KoSimpleQA (CC-BY-4.0), at the commit this task is
# pinned to. ``kosimpleqa.json`` is the 1,000-question set of the paper.
REPO_URL = "https://raw.githubusercontent.com/naver-ai/KoSimpleQA"
DATA_FILE = "kosimpleqa.json"
DATASET_REVISION = "1bdbb613c39973538239a161ea2425d763b25fe8"


def record_to_sample(record: dict) -> Sample:
    """One question as a single user turn, with nothing around it (as simple-evals sends it)."""
    return Sample(input=record["content"], target=record["answer"],
                  metadata={"question": record["content"], "category": record["category"]})


@task
def kosimpleqa(thinking: bool = True, epochs: int = 1, judge_retries: int = 2,
               revision: str = DATASET_REVISION) -> Task:
    """KoSimpleQA, 1,000 questions x ``epochs`` runs; needs a judge bound to the ``grader`` role.

    KoSimpleQA (Ko et al., 2025, arXiv:2510.18368) follows SimpleQA: each answer is graded
    CORRECT, INCORRECT or NOT_ATTEMPTED, and the paper reports correct (CO), not attempted
    (NA), incorrect (IN), correct given attempted (CGA) and the F-score. The paper does not
    publish its grader; this task uses the SimpleQA grader of OpenAI's simple-evals
    (``scorers/simpleqa_grade.py``), whose evaluation code the paper says it follows.

    Reported: ``accuracy`` (= CO, the headline), ``simpleqa_metrics`` (CO, IN, NA, CGA,
    F-score), ``accuracy`` per category (``category/<name>``), ``answer_diagnostics`` and
    ``judge_failures``.

    Deviations:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the paper: temperature 1.0, top-p 1.0, 2,048 tokens);
    - the grader is the model bound to the ``grader`` role, called with simple-evals' system
      message and temperature but without its 2,048-token cap, so a reasoning judge is not cut
      off;
    - a truncated or empty answer is graded NOT_ATTEMPTED without asking the grader, as the
      paper does for a reasoning model that runs out of tokens;
    - the questions are read from the authors' GitHub release at a pinned commit.
    """
    return Task(
        dataset=json_dataset(f"{REPO_URL}/{revision}/{DATA_FILE}", sample_fields=record_to_sample,
                             auto_id=True, name="kosimpleqa"),
        solver=generate(),
        scorer=simpleqa_grade(judge_retries=judge_retries),
        metrics=[accuracy(), stderr(), simpleqa_metrics(), answer_diagnostics(), judge_failures(),
                 grouped(accuracy(), "category", all=False, name_template="category/{group_name}")],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
