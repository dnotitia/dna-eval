"""Hallucination: false premises, nonexistent entities, unknowable facts, and grounding
(``dnotitia/dna_hallucination``)."""
from inspect_ai import Task, task
from inspect_ai.scorer import mean

from dna_eval.dataset import dna_dataset
from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate
from dna_eval.tasks import HEADLINE, SUITE_VERSION, dna_generate_config

# dnotitia/dna_hallucination commit this task is pinned to.
DATASET_REVISION = "1d200f984559f4a06cd871e6d4dde792ff5149d1"


@task
def hallucination(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    """100 prompts in 5 categories of 20 x ``epochs`` runs, judged once per answer.

    Besides the overall per-run statistics, each category (``false_premise``,
    ``true_premise``, ``nonexistent``, ``unknowable``, ``grounding``) gets its own
    ``<category>/trimmed_mean`` and ``<category>/trimmed_passed`` from the same judgments.
    """
    return Task(
        dataset=dna_dataset("dna_hallucination", revision=revision,
                            metadata=["category", "pair_id", "difficulty", "mode"]),
        solver=dna_generate(),
        scorer=rubric_scorer("hallucination"),
        metrics=[per_run_trimmed(), per_run_trimmed(group_by="category"), mean()],
        epochs=epochs,
        config=dna_generate_config(thinking),
        headline_metric=HEADLINE,
        version=SUITE_VERSION,
    )
