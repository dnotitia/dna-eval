"""Style: does the model reason and answer in the style of a reference model? (``dnotitia/dna_style``)

Each run's reasoning and answer are compared, for style only, with a fixed reference
response from another model, typically the same-size base model the evaluated model was
trained from. See ``docs/running-style-task.md``.

References are published with the dataset as ``references/<model>.parquet``. For a model
without published references, bind it to the ``reference`` role: its responses are generated
once per sample and cached locally (``solvers/style_reference.py``).
"""
from inspect_ai import Task, task

from dna_eval.dataset import dna_dataset, dna_files, dna_table
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate
from dna_eval.solvers.style_reference import NO_REFERENCE_ANSWER, NO_REFERENCE_REASONING, style_reference
from dna_eval.tasks.dna_suite import HEADLINE, dna_generate_config

# dnotitia/dna_style commit this task is pinned to.
DATASET_REVISION = "81e1e21ba8705691c6340598c4aeb140ae901c51"

JUDGES = ("reasoning_style", "content_style")
REFERENCE_SOURCES = ("auto", "hub", "generate")


def published_references(revision: str) -> list[str]:
    """Reference models published with ``dnotitia/dna_style`` at ``revision``."""
    return sorted(f.removeprefix("references/").removesuffix(".parquet")
                  for f in dna_files("dna_style", revision=revision)
                  if f.startswith("references/") and f.endswith(".parquet"))


@task
def style(reference: str, thinking: bool = True, epochs: int = 15, revision: str = DATASET_REVISION,
          judges: str | list[str] = JUDGES, reference_source: str = "auto",
          refresh_reference: bool = False) -> Task:
    """50 prompts x ``epochs`` runs, one answer per run judged by each rubric in ``judges``.

    ``reference`` names the reference model. It has no default: the comparison is only
    meaningful against the right model, usually the base of the model under evaluation.

    ``reference_source``:
      ``auto``     - published references if there are any for ``reference``, otherwise
                     generate them with the ``reference`` role (cached locally);
      ``hub``      - published references only;
      ``generate`` - always use the ``reference`` role (and its local cache).
    ``refresh_reference`` regenerates instead of reading the local cache.
    """
    selected = [judges] if isinstance(judges, str) else list(judges)
    unknown = sorted(set(selected) - set(JUDGES))
    if unknown:
        raise ValueError(f"unknown judges {unknown}; choose from {list(JUDGES)}")
    if reference_source not in REFERENCE_SOURCES:
        raise ValueError(f"reference_source must be one of {list(REFERENCE_SOURCES)}")

    published = published_references(revision)
    use_hub = reference_source != "generate" and reference in published
    if reference_source == "hub" and not use_hub:
        raise ValueError(f"no published reference {reference!r} in dnotitia/dna_style@{revision[:7]}; "
                         f"published: {published}. Use reference_source=auto or generate with the "
                         f"'reference' model role to generate references.")

    dataset = dna_dataset("dna_style", revision=revision, metadata=["source", "related_topic", "relation", "lang"])
    if use_hub:
        refs = {r["id"]: r for r in dna_table("dna_style", f"references/{reference}.parquet", revision=revision)}
        for sample in dataset:
            ref = refs.get(sample.id, {})
            sample.metadata = {
                **(sample.metadata or {}),
                "reference_reasoning": (ref.get("reference_reasoning") or "").strip() or NO_REFERENCE_REASONING,
                "reference_content": (ref.get("reference_content") or "").strip() or NO_REFERENCE_ANSWER,
            }

    config = dna_generate_config(thinking)
    return Task(
        dataset=dataset,
        solver=[style_reference(reference, revision, config=config, refresh=refresh_reference), dna_generate()],
        scorer=[rubric_scorer(j) for j in selected],
        epochs=epochs,
        config=config,
        headline_metric=HEADLINE,
        metadata={"reference": reference, "reference_source": "hub" if use_hub else "local"},
    )
