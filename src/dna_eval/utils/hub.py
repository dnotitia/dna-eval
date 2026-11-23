"""Read the data files of a Hugging Face dataset directly, at a pinned revision.

``datasets.load_dataset`` resolves the dataset on the Hub again for every config, even when the
files are cached, which takes about a second per call; a dataset split into dozens of configs
takes a minute to load that way. Here the dataset card says which file holds each config's
split, and ``hf_hub_download`` returns a file cached for a commit revision without a request.
"""
from __future__ import annotations

import csv
import functools
import json

from huggingface_hub import hf_hub_download
from huggingface_hub.repocard import metadata_load


@functools.cache
def config_files(repo: str, revision: str) -> dict[tuple[str, str], str]:
    """The file of each (config, split), as the ``configs`` of the dataset card at ``revision``
    map them."""
    card = hf_hub_download(repo, "README.md", repo_type="dataset", revision=revision)
    return {(c["config_name"], f["split"]): f["path"] for c in metadata_load(card)["configs"]
            for f in c["data_files"]}


def read_config(repo: str, config: str, split: str, revision: str) -> list[dict]:
    """The rows of a config's ``split`` at ``revision``: a CSV file (every value a string) or a
    JSON file (a list of records)."""
    path = hf_hub_download(repo, config_files(repo, revision)[(config, split)], repo_type="dataset",
                           revision=revision)
    with open(path, encoding="utf-8", newline="") as f:
        if path.endswith(".csv"):
            return list(csv.DictReader(f))
        if path.endswith(".json"):
            return json.load(f)
    raise ValueError(f"unsupported data file {path!r}: expected .csv or .json")
