"""Shortcut loader for the suite's own ``dnotitia/dna_*`` datasets.

These share one schema: ``id``, ``input``, optional ``target``, and free metadata
columns. It is a convenience, not a requirement: a task may build its dataset with any
Inspect loader (``hf_dataset``, ``json_dataset``, a custom ``record_to_sample``), since
``rubric_scorer`` only reads the sample's input, target and ``metadata["context"]``.
"""
from inspect_ai.dataset import Dataset, FieldSpec, hf_dataset

HF_ORG = "dnotitia"


def dna_dataset(name: str, *, revision: str,
                metadata: list[str] | None = None, split: str = "train") -> Dataset:
    """Load ``dnotitia/<name>`` at a Hub commit as an Inspect dataset.

    ``revision`` is required: each task pins it as the default of its own ``revision``
    argument, so the commit is recorded in the log's ``task_args`` and a dataset update
    cannot silently move a reported number. Authentication for private repos follows
    ``huggingface_hub`` (``HF_TOKEN`` or a cached login); the token is not passed explicitly.
    """
    return hf_dataset(
        f"{HF_ORG}/{name}",
        split=split,
        revision=revision,
        sample_fields=FieldSpec(id="id", input="input", target="target", metadata=metadata or []),
    )
