"""AIME 2026: the 30 problems of AIME I and II 2026, evaluated as MathArena does (see ``aime.py``)."""
from inspect_ai import Task, task

from dna_eval.tasks.benchmarks.aime import aime_task

# MathArena/aime_2026 commit this task is pinned to.
DATASET_REVISION = "d2de22f3c656b4f56cf8981212186377d1e23bc3"


@task
def aime2026(thinking: bool = True, epochs: int = 8, revision: str = DATASET_REVISION) -> Task:
    """30 problems x ``epochs`` runs; the headline is ``accuracy`` (pass@1 averaged over runs)."""
    return aime_task(2026, revision, thinking, epochs)
