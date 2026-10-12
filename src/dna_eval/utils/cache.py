"""The local cache directory, and files downloaded once and pinned by checksum."""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

CACHE_ENV = "DNA_EVAL_CACHE_DIR"


def cache_root() -> Path:
    """``$DNA_EVAL_CACHE_DIR``, or ``~/.cache/dna_eval``."""
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "dna_eval")


def fetch_pinned(url: str, path: Path, sha256: str | None) -> Path:
    """``path``, downloaded from ``url`` on first use.

    The bytes are checked against ``sha256`` both when downloaded and when read from the
    cache; a mismatch raises ``ValueError`` and a bad download is not kept. ``sha256=None``
    skips the check.
    """
    cached = path.exists()
    if cached:
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    if sha256 is not None:
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            where = path if cached else url
            raise ValueError(f"{where} has SHA-256 {actual}, expected {sha256}")
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return path
