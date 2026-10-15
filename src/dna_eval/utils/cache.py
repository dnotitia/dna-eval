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


_CHUNK = 1 << 20


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_pinned(url: str, path: Path, sha256: str | None) -> Path:
    """``path``, downloaded from ``url`` on first use.

    The bytes are checked against ``sha256`` both when downloaded and when read from the
    cache; a mismatch raises ``ValueError`` and a bad download is not kept. ``sha256=None``
    skips the check. The file is streamed to disk and hashed in chunks, so its size is not
    limited by memory.
    """
    if path.exists():
        if sha256 is not None and (actual := _sha256_of(path)) != sha256:
            raise ValueError(f"{path} has SHA-256 {actual}, expected {sha256}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
            while chunk := response.read(_CHUNK):
                digest.update(chunk)
                out.write(chunk)
        if sha256 is not None and (actual := digest.hexdigest()) != sha256:
            raise ValueError(f"{url} has SHA-256 {actual}, expected {sha256}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
