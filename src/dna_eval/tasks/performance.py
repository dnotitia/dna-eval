"""Performance: general Korean QA against reference answers (``dnotitia/dna_performance``)."""
from inspect_ai import Task, task
from inspect_ai.solver import generate

from dna_eval.dataset import dna_dataset
from dna_eval.generation import dna_generate_config
from dna_eval.scorers.rubric_scorer import rubric_scorer

# dnotitia/dna_performance commit used for the published baselines.
DATASET_REVISION = "d4582ca540eb8f6b7369b5e7853626acb87b603d"


@task
def performance(thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION) -> Task:
    """30 prompts x ``epochs`` runs. Official numbers: ``per_run_trimmed`` (trimmed mean + pass count).

    ``thinking`` is forwarded as ``chat_template_kwargs.enable_thinking`` (vLLM / Qwen-style
    templates). Sampling is left to the served model's ``generation_config``.
    """
    return Task(
        dataset=dna_dataset("dna_performance", revision=revision),
        solver=generate(),
        scorer=rubric_scorer("performance"),
        epochs=epochs,
        config=dna_generate_config(thinking),
    )
