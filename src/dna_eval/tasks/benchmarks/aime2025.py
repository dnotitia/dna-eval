"""AIME 2025: the 30 problems of AIME I and II 2025, evaluated as MathArena does (see ``aime.py``)."""
from inspect_ai import Task, task

from dna_eval.tasks.benchmarks.aime import aime_task

# MathArena/aime_2025 commit this task is pinned to.
DATASET_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"


@task
def aime2025(thinking: bool = True, epochs: int = 8, revision: str = DATASET_REVISION) -> Task:
    """30 problems x ``epochs`` runs; the headline is ``accuracy`` (pass@1 averaged over runs)."""
    return aime_task(2025, revision, thinking, epochs)
