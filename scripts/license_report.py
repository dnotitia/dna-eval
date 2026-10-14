"""List the data every task uses, with its licence and access terms.

    python scripts/license_report.py            # markdown table (needs network for licences)
    python scripts/license_report.py --offline  # only which data each task uses
    python scripts/license_report.py --json     # the discovered data, as JSON

A lookup aid for contributors and reviewers (see "Before you start: licences" in
docs/adding-a-task.md); it checks nothing and enforces nothing. Nothing has to be declared:
every registered task is built once with the data loaders replaced by stubs that record what
they were asked for (a Hub dataset and revision, or a pinned URL) and stop. No data is
downloaded: connections are refused while tasks are built, so a task that loads data another
way shows "network access" or "no data found" instead of downloading. Licences come from the
Hub dataset card at the pinned revision, gated-access terms included.

Run it as a script, in its own process: the stubs must be in place before dna_eval is imported.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import pkgutil
import re
import socket
import sys
from dataclasses import asdict, dataclass, field
from importlib import import_module

PLACEHOLDER = "placeholder"   # value for task arguments with no default (e.g. style's reference)


@dataclass
class Source:
    kind: str                 # "hub" (a Hub dataset) or "url" (a pinned download)
    name: str                 # dataset id, or URL
    revision: str | None = None


@dataclass
class TaskData:
    task: str
    module: str
    sources: list[Source] = field(default_factory=list)
    error: str = ""


class _Stop(Exception):
    """Raised by a stub once it has recorded what was asked for."""


_seen: list[Source] = []


def _record(source: Source):
    _seen.append(source)
    raise _Stop


def _install_stubs() -> None:
    import datasets
    import huggingface_hub
    import inspect_ai.dataset

    inspect_ai.dataset.hf_dataset = lambda path, *a, revision=None, **k: _record(Source("hub", path, revision))
    huggingface_hub.hf_hub_download = (
        lambda repo_id, *a, revision=None, **k: _record(Source("hub", repo_id, revision)))
    huggingface_hub.HfApi.list_repo_files = (
        lambda self, repo_id, *a, revision=None, **k: _record(Source("hub", repo_id, revision)))
    datasets.load_dataset = lambda path, *a, revision=None, **k: _record(Source("hub", path, revision))

    import dna_eval.utils.cache as cache
    cache.fetch_pinned = lambda url, path, sha256: _record(Source("url", url, sha256))


class NetworkBlocked(Exception):
    """Raised for any connection attempt while tasks are being built."""


@contextlib.contextmanager
def _no_network():
    """Refuse every outgoing connection inside the block.

    Sockets are refused rather than setting ``HF_HUB_OFFLINE``, which the Hub client reads once
    at import and would keep for the licence lookups that follow.
    """
    def refuse(*a, **k):
        raise NetworkBlocked("network access while building the task (a data loader this report does not stub?)")

    saved = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    socket.socket.connect = socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = saved


def discover() -> list[TaskData]:
    """Build every task under dna_eval.tasks once, offline, and record the data it asks for."""
    with _no_network():
        return _discover()


def _discover() -> list[TaskData]:
    _install_stubs()
    from inspect_ai._util.registry import is_registry_object, registry_info

    import dna_eval.tasks

    found: list[TaskData] = []
    for mod in pkgutil.walk_packages(dna_eval.tasks.__path__, "dna_eval.tasks."):
        module = import_module(mod.name)
        for obj in vars(module).values():
            if not (callable(obj) and is_registry_object(obj, type="task")
                    and getattr(obj, "__module__", "") == module.__name__):
                continue
            name = registry_info(obj).name.split("/")[-1]
            args = {p.name: PLACEHOLDER for p in inspect.signature(obj).parameters.values()
                    if p.default is inspect.Parameter.empty}
            _seen.clear()
            data = TaskData(name, module.__name__)
            try:
                obj(**args)
            except _Stop:
                pass
            except Exception as e:  # noqa: BLE001  (reported, not fatal)
                data.error = f"{type(e).__name__}: {e}"[:200]
            data.sources = list(_seen)
            found.append(data)
    return sorted(found, key=lambda d: d.task)


def hub_licence(source: Source) -> dict:
    """Licence and access terms of a Hub dataset at its pinned revision."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(source.name, revision=source.revision)
    except Exception as e:  # noqa: BLE001
        return {"licence": None, "terms": "", "failed": f"lookup failed: {type(e).__name__}"}
    card = info.card_data or {}
    get = card.get if hasattr(card, "get") else (lambda k: None)
    terms = []
    if info.gated:
        terms.append("gated")
    if get("extra_gated_prompt"):
        terms.append(" ".join(str(get("extra_gated_prompt")).split())[:160])
    fields = get("extra_gated_fields") or {}
    terms += [k for k in fields if re.search(r"commercial|agree", k, re.I)]
    licence = get("license")
    return {"licence": ", ".join(licence) if isinstance(licence, list) else licence, "terms": "; ".join(terms)}


def report(found: list[TaskData], offline: bool) -> str:
    """Markdown table: one row per task and data source."""
    rows = []
    for d in found:
        notes = [] if d.sources else ["no data found"]
        for s in d.sources or [Source("none", "—")]:
            looked = hub_licence(s) if not offline and s.kind == "hub" else {"licence": None, "terms": "", "failed": ""}
            if looked.get("failed"):
                notes.append(looked["failed"])
            if d.error:
                notes.append(f"build failed: {d.error}")
            source = f"{s.name}@{(s.revision or '')[:7]}" if s.kind == "hub" else s.name
            licence = ("—" if offline or s.kind != "hub" else "unavailable" if looked.get("failed")
                       else looked["licence"] or "none stated")
            rows.append(f"| `{d.task}` | {source} | {licence} | {looked['terms']} | {'; '.join(notes)} |")
            notes = []
    note = ("Data that is not on the Hub (a URL): read its licence where it is published. \"no data found\" "
            "or \"network access\": the task loads data in a way this script does not see; load it with "
            "`hf_dataset` or `fetch_pinned`, or teach the script the loader.")
    return "\n".join(["| task | data | licence | access terms | note |", "| --- | --- | --- | --- | --- |",
                      *rows, "", note, ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="do not look licences up on the Hub")
    parser.add_argument("--json", action="store_true", help="print the discovered data as JSON")
    args = parser.parse_args()
    found = discover()
    print(json.dumps([asdict(d) for d in found], indent=1) if args.json else report(found, args.offline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
