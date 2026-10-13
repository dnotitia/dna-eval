"""Report the data each task uses and its licence, against the table in docs/benchmarks.md.

    python scripts/license_report.py            # markdown report (needs network for licences)
    python scripts/license_report.py --offline  # only which data each task uses
    python scripts/license_report.py --strict   # exit 1 if a benchmark is missing from, or
                                                # disagrees with, the licence table

Nothing has to be declared for this: every registered task is built once with the data
loaders replaced by stubs that record what they were asked for (a Hub dataset and revision,
or a pinned URL) and stop. No data is downloaded. Licences come from the Hub dataset card at
the pinned revision, gated-access terms included.

The report is advice, not a verdict: a non-commercial or missing licence is for a maintainer
to decide on (see "Licences" in docs/adding-a-task.md). Run it as a script, in its own
process: the stubs must be in place before dna_eval is imported.
"""
from __future__ import annotations

import argparse
import inspect
import json
import pkgutil
import re
import sys
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_DOC = ROOT / "docs" / "benchmarks.md"
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


def discover() -> list[TaskData]:
    """Build every task under dna_eval.tasks once and record the data it asks for."""
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


def table_licences() -> dict[str, str]:
    """task -> licence as written in the "Data licences" table of docs/benchmarks.md."""
    text = TABLE_DOC.read_text(encoding="utf-8")
    section = text.split("## Data licences", 1)[-1].split("\n## ", 1)[0]
    out = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and "`" in cells[0]:
            for task in re.findall(r"`([^`]+)`", cells[0]):
                out[task] = cells[1]
    return out


def _norm(licence: str | None) -> str:
    return "none" if not licence or licence.lower().startswith("none") else licence.lower()


def report(found: list[TaskData], offline: bool) -> tuple[str, int]:
    """Markdown report and the number of benchmark tasks that need attention."""
    table = table_licences()
    rows, attention = [], 0
    for d in found:
        own = ".dna_suite." in d.module
        for s in d.sources or [Source("none", "—")]:
            looked = {"licence": None, "terms": "", "failed": ""}
            if not offline and s.kind == "hub":
                looked = hub_licence(s)
            status = "suite data" if own else ""
            if not own:
                listed = table.get(d.task)
                card = _norm(looked["licence"])
                if listed is None:
                    status, needs = "not in the table", True
                elif offline or s.kind != "hub" or looked.get("failed"):
                    status, needs = "ok", False
                elif card == "none" and _norm(listed) != "none":
                    status, needs = f"the card states none; the table says {listed} (from the dataset's README?)", False
                elif card != _norm(listed):
                    status, needs = f"the table says {listed}", True
                else:
                    status, needs = "ok", False
                if needs or d.error:
                    attention += 1
            if d.error:
                status += f"; build failed: {d.error}"
            source = f"{s.name}@{(s.revision or '')[:7]}" if s.kind == "hub" else s.name
            licence = ("—" if offline or s.kind != "hub" else "unavailable" if looked.get("failed")
                       else looked["licence"] or "none stated")
            if looked.get("failed"):
                status += f" ({looked['failed']})"
            rows.append(f"| `{d.task}` | {source} | {licence} | {looked['terms']} | {status} |")
    head = ["| task | data | licence | access terms | check |", "| --- | --- | --- | --- | --- |"]
    note = ("Licences of non-Hub files (`url`) are in docs/benchmarks.md. A non-commercial or missing "
            "licence needs a maintainer's decision.")
    return "\n".join(["## Data licences", "", *head, *rows, "", note, ""]), attention


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="do not look licences up on the Hub")
    parser.add_argument("--strict", action="store_true", help="exit 1 when a benchmark needs attention")
    parser.add_argument("--json", action="store_true", help="print the discovered data as JSON")
    args = parser.parse_args()
    found = discover()
    if args.json:
        print(json.dumps([asdict(d) for d in found], indent=1))
        return 0
    text, attention = report(found, args.offline)
    print(text)
    return 1 if args.strict and attention else 0


if __name__ == "__main__":
    sys.exit(main())
