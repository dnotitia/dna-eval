"""Uncensorship: does the model engage with topics Chinese-origin base models refuse? (``dnotitia/dna_uncensorship``)"""
from inspect_ai import Task, task

from dna_eval.dataset import dna_dataset
from dna_eval.generation import dna_generate_config
from dna_eval.scorers.composite_scorer import composite_scorer
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate

# dnotitia/dna_uncensorship commit this task is pinned to.
DATASET_REVISION = "1fb5ebd1dc245d7f2d85f6853015279f0a776a0a"


@task
def uncensorship(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    """50 prompts x ``epochs`` runs. Two judge passes per response (government bias, response
    quality); the sample score is their mean and a sample passes only if both clear 0.7."""
    return Task(
        dataset=dna_dataset("dna_uncensorship", revision=revision),
        solver=dna_generate(),
        scorer=composite_scorer(
            scorers=[rubric_scorer("uncensorship_bias"), rubric_scorer("uncensorship_quality")],
            names=["bias", "quality"],
        ),
        epochs=epochs,
        config=dna_generate_config(thinking),
    )