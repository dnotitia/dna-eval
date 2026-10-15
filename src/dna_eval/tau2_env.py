"""Access to the tau-bench environments (tau2-bench, "tau^3") shared by the tau-bench solver,
scorer and task.

tau2-bench (github.com/sierra-research/tau2-bench, MIT, Copyright (c) 2025 Sierra Research) is
the benchmark's own code: environments, tools, tasks and the evaluator. It is not on PyPI under
that name, and its wheel carries no data, so this module

- installs nothing by itself but names the install command (``INSTALL_COMMAND``);
- downloads the repository archive at the pinned commit once, checks its SHA-256 and extracts
  only the data the four text domains and the user simulator read into the cache directory
  (``utils/cache.py``), then points ``TAU2_DATA_DIR`` at it. tau2 reads that variable when it is
  first imported, so ``tau2_ready`` must run before anything imports ``tau2``; a ``TAU2_DATA_DIR``
  the user already set is used as it is and nothing is downloaded;
- keeps tau2's loguru output at ``ERROR`` (as the official CLI does by default) before importing it;
- loads and caches a domain's tasks and builds the ``env_kwargs`` the environment constructor and
  the evaluator need (``banking_knowledge`` takes a retrieval variant).

The ``state.store`` keys the solver writes and the scorer reads are defined here, so neither
imports the other (decision record 0011).
"""
from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from dna_eval.utils.cache import cache_root, fetch_pinned

# tau2-bench v1.0.1 (tag), as the immutable commit the tag points to (decision record 0008).
TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU2_ARCHIVE_URL = f"https://codeload.github.com/sierra-research/tau2-bench/tar.gz/{TAU2_COMMIT}"
TAU2_ARCHIVE_SHA256 = "7a227036d07fdeb088dbd0c99fdb40ee9b9e0ebc06aadbc4dec1217b3383e57e"

# tau2 needs Python 3.12 or 3.13. It imports the standard-library ``audioop`` (removed in 3.13)
# without declaring the ``audioop-lts`` backport, hence the second requirement.
INSTALL_COMMAND = (f'pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench@{TAU2_COMMIT}" '
                   '"audioop-lts; python_version >= \'3.13\'"')

DOMAINS = ("airline", "retail", "telecom", "banking_knowledge")

# ``banking_knowledge`` retrieval variants that run offline (no embedding or reranker API, no
# sandbox shell): the knowledge base is searched with BM25 (needs ``rank-bm25``), grepped,
# inlined in the policy, inlined for the task's golden documents, or absent.
OFFLINE_RETRIEVAL_VARIANTS = ("bm25", "bm25_grep", "grep_only", "full_kb", "golden_retrieval", "no_knowledge")
BM25_VARIANTS = ("bm25", "bm25_grep")

# ``state.store`` keys: the solver writes them, the scorer reads them (``limit_reached`` and
# ``turns`` are the keys ``metrics/agent_diagnostics.py`` defines and are written there too).
SIMULATION_KEY = "tau_simulation"      # the tau2 SimulationRun, as JSON (``model_dump(mode="json")``)
TERMINATION_KEY = "tau_termination"    # ``TerminationReason`` value the conversation ended with

_DATA_PARTS = ("domains", "user_simulator")     # directories of data/tau2/ that are extracted
_ready = False


def _extract_data(archive: Path, dest: Path) -> None:
    """Extract ``data/tau2/{domains,user_simulator}`` of the archive into ``dest`` (a new directory)
    as ``dest/data/...``. Members are checked by tarfile's ``data`` filter."""
    prefix = f"tau2-bench-{TAU2_COMMIT}"
    with tarfile.open(archive) as tar:
        wanted = [m for m in tar.getmembers()
                  if (parts := Path(m.name).parts)[:1] == (prefix,) and parts[1:3] == ("data", "tau2")
                  and (len(parts) == 3 or parts[3] in _DATA_PARTS)]
        tar.extractall(dest, members=wanted, filter="data")


def ensure_data() -> Path:
    """The tau2 data directory (``TAU2_DATA_DIR``), downloaded and extracted on first use.

    The archive is fetched at the pinned commit and checked against ``TAU2_ARCHIVE_SHA256``. The
    extraction happens in a new temporary directory and is moved into place in one rename, so an
    interrupted run leaves no half-extracted data. Sets ``TAU2_DATA_DIR`` for this process.
    """
    given = os.environ.get("TAU2_DATA_DIR")
    if given:
        return Path(given)
    root = cache_root() / "tau2"
    data = root / TAU2_COMMIT / "data"
    if not data.is_dir():
        archive = fetch_pinned(TAU2_ARCHIVE_URL, root / f"tau2-bench-{TAU2_COMMIT}.tar.gz", TAU2_ARCHIVE_SHA256)
        root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=root, prefix="extract-"))
        try:
            _extract_data(archive, tmp)
            (root / TAU2_COMMIT).mkdir(exist_ok=True)
            try:
                (tmp / f"tau2-bench-{TAU2_COMMIT}" / "data").rename(data)
            except OSError:       # another process finished first
                if not data.is_dir():
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        archive.unlink(missing_ok=True)
    os.environ["TAU2_DATA_DIR"] = str(data)
    return data


def tau2_ready() -> None:
    """Make ``import tau2`` safe: point it at the data, quieten its logging. Idempotent.

    Raises ``RuntimeError`` with the install command when tau2 is missing, or when tau2 was
    imported before with another data directory (it fixes the directory at import time).
    """
    global _ready
    if _ready:
        return
    if sys.version_info < (3, 12) or importlib.util.find_spec("tau2") is None:
        raise RuntimeError(f"tau-bench needs Python 3.12 or 3.13 and the 'tau2' package: {INSTALL_COMMAND}")
    data = ensure_data()
    if "tau2" not in sys.modules:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    from tau2.utils.utils import DATA_DIR
    if Path(DATA_DIR) != data:
        raise RuntimeError(f"tau2 was imported with data directory {DATA_DIR} before dna-eval set "
                           f"TAU2_DATA_DIR={data}; set TAU2_DATA_DIR before importing tau2")
    _ready = True


def check_retrieval_variant(domain: str, variant: str) -> None:
    """Raise ``ValueError``/``RuntimeError`` for a ``banking_knowledge`` retrieval variant that
    cannot run here: not offline, or BM25 without ``rank-bm25``. Other domains ignore the variant."""
    if domain != "banking_knowledge":
        return
    if variant not in OFFLINE_RETRIEVAL_VARIANTS:
        raise ValueError(f"retrieval_variant {variant!r} is not supported: it needs an embedding or "
                         f"reranker API or a sandbox shell. Use one of {', '.join(OFFLINE_RETRIEVAL_VARIANTS)}")
    if variant in BM25_VARIANTS and importlib.util.find_spec("rank_bm25") is None:
        raise RuntimeError(f"retrieval_variant {variant!r} needs the 'rank-bm25' package: pip install rank-bm25")


@functools.lru_cache(maxsize=None)
def _tasks(domain: str, split: str) -> dict[str, Any]:
    from tau2.registry import registry
    tasks = registry.get_tasks_loader(domain)(task_split_name=split)
    return {t.id: t for t in tasks}


def load_tasks(domain: str, split: str = "base") -> list:
    """The tau2 ``Task`` objects of ``domain`` in ``split``, in the official order."""
    tau2_ready()
    return list(_tasks(domain, split).values())


def get_task(domain: str, split: str, task_id: str):
    """The tau2 ``Task`` with this id."""
    tau2_ready()
    return _tasks(domain, split)[task_id]


def env_kwargs(domain: str, retrieval_variant: str | None, task) -> dict:
    """The keyword arguments the environment constructor and ``evaluate_simulation`` take.

    As tau2's ``runner/build.py`` builds them: empty except for ``banking_knowledge``, which gets
    the retrieval variant, the task (the golden-retrieval variant inlines its documents) and the
    set of read-only discoverable tools whose calls are logged for grading.
    """
    if domain != "banking_knowledge":
        return {}
    from tau2.runner.build import _derive_read_log_allowlist
    return {"retrieval_variant": retrieval_variant, "task": task,
            "read_log_allowlist": _derive_read_log_allowlist(task)}
