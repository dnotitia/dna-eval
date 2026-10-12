"""Persona: does the model describe itself and its maker correctly? (``dnotitia/dna_persona``)"""
from inspect_ai import Task, task

from dna_eval.dataset import dna_dataset, dna_file
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate
from dna_eval.tasks.dna_suite import HEADLINE, dna_generate_config

# dnotitia/dna_persona commit this task is pinned to.
DATASET_REVISION = "1f8dc293fc184f110d1d7d154622e1c137730698"


@task
def persona(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    """50 questions about the model's identity x ``epochs`` runs.

    The model gets only the question, no system prompt: the task measures what the model
    knows about itself. The persona sheet (``persona.txt`` in the dataset repo) is shown to
    the judge as ``context``, the reference the answer is checked against.
    """
    sheet = dna_file("dna_persona", "persona.txt", revision=revision).strip()
    return Task(
        dataset=dna_dataset("dna_persona", revision=revision),
        solver=dna_generate(),
        scorer=rubric_scorer("persona", context=[sheet]),
        epochs=epochs,
        config=dna_generate_config(thinking),
        headline_metric=HEADLINE,
    )
