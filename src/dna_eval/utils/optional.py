"""Optional dependencies: fail early, with the command that installs them."""
from __future__ import annotations

import importlib.util
import sys

# Installs what the agent tasks need from PyPI (``inspect-harbor``, ``inspect-swe``).
AGENTIC_INSTALL = 'pip install "dna-eval[agentic]"'


def require(module: str, *, python: tuple[int, int] = (3, 10), install: str) -> None:
    """Raise a ``RuntimeError`` naming ``install`` when ``module`` cannot be imported, or one
    naming the Python version when the running Python is older than ``python``."""
    if sys.version_info < python:
        raise RuntimeError(f"this task needs Python {python[0]}.{python[1]} or later "
                           f"(running {sys.version_info.major}.{sys.version_info.minor})")
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"this task needs the '{module}' package: {install}")
