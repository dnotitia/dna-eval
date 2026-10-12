"""Degeneration: the answer is right but the output breaks down (``dnotitia/dna_confusion``).

Two failure modes, judged independently on the same answer: language confusion (foreign
scripts or English leaking into Korean prose) and repetition (loops, restated content,
output that never terminates).
"""
from inspect_ai import Task, task

from dna_eval.dataset import dna_dataset
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate
from dna_eval.tasks.dna_suite import HEADLINE, dna_generate_config

# dnotitia/dna_confusion commit this task is pinned to.
DATASET_REVISION = "674bb8e0fb134cfb9d001dff54b387d6c9bb9a3a"

JUDGES = ("language_confusion", "repetition")


@task
def degeneration(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION,
                 judges: str | list[str] = JUDGES) -> Task:
    """45 Korean prompts x ``epochs`` runs, one answer per run judged by each rubric in ``judges``.

    Each judge is its own scorer with its own per-run statistics; they are not combined.
    ``-T judges=repetition`` runs a single one.
    """
    selected = [judges] if isinstance(judges, str) else list(judges)
    unknown = sorted(set(selected) - set(JUDGES))
    if unknown:
        raise ValueError(f"unknown judges {unknown}; choose from {list(JUDGES)}")
    return Task(
        dataset=dna_dataset("dna_confusion", revision=revision),
        solver=dna_generate(),
        scorer=[rubric_scorer(j) for j in selected],
        epochs=epochs,
        config=dna_generate_config(thinking),
        headline_metric=HEADLINE,
    )
