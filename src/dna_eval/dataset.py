"""Shortcut loader for the suite's own ``dnotitia/dna_*`` datasets.

These share one schema: ``id``, ``input``, optional ``target``, and free metadata
columns. It is a convenience, not a requirement: a task may build its dataset with any
Inspect loader (``hf_dataset``, ``json_dataset``, a custom ``record_to_sample``), since
``rubric_scorer`` only reads the sample's input, target and ``metadata["context"]``.
"""
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
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


def dna_file(name: str, filename: str, *, revision: str) -> str:
    """Text of a file stored next to the data in ``dnotitia/<name>``, at the same pinned commit."""
    path = hf_hub_download(f"{HF_ORG}/{name}", filename, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        return f.read()


def dna_files(name: str, *, revision: str) -> list[str]:
    """Paths of the files in ``dnotitia/<name>`` at the pinned commit."""
    return HfApi().list_repo_files(f"{HF_ORG}/{name}", repo_type="dataset", revision=revision)


def dna_table(name: str, filename: str, *, revision: str) -> list[dict]:
    """Rows of a parquet file stored next to the data in ``dnotitia/<name>``, at the same commit."""
    path = hf_hub_download(f"{HF_ORG}/{name}", filename, repo_type="dataset", revision=revision)
    return pd.read_parquet(path).to_dict("records")
