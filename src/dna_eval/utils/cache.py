"""The local cache directory."""
from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV = "DNA_EVAL_CACHE_DIR"


def cache_root() -> Path:
    """``$DNA_EVAL_CACHE_DIR``, or ``~/.cache/dna_eval``."""
    return Path(os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "dna_eval")
