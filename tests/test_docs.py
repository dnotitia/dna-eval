"""The documents people and agents follow stay true to the repository.

- Every repository path a document names (in backticks or as a relative link) exists, except
  the docs' hypothetical example task (names starting with ``my_``).
- CLAUDE.md imports only files that exist.
- The decision index lists every record with the record's own title and status.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "dna_eval"
DECISIONS = ROOT / "docs" / "decisions"
DOCS = sorted([ROOT / n for n in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md")]
              + list((ROOT / "docs").rglob("*.md")))
ROOT_FILES = {"README.md", "CONTRIBUTING.md", "AGENTS.md", "CLAUDE.md", "pyproject.toml"}
PATHLIKE = re.compile(r"^[\w.\-/]+(\.(py|md|yaml|jinja|json|txt|toml)|/)$")
STATUS = re.compile(r"^accepted$|^proposed$|^deprecated$|^superseded by (\d{4})$")


def _named_paths(doc: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)   # code blocks are examples
    names = re.findall(r"`([^`\s]+)`", text) + re.findall(r"\]\(([^)\s#]+)\)", text)
    return [n for n in names if not n.startswith(("http", "$", "~")) and PATHLIKE.match(n)
            and ("/" in n or n in ROOT_FILES) and not Path(n).name.startswith("my_")]


def _exists(name: str, doc: Path) -> bool:
    bases = (doc.parent, ROOT, PACKAGE)        # relative to the document, the repo, or the package
    return any((base / name).exists() for base in bases)


@pytest.mark.parametrize("doc", DOCS, ids=[str(d.relative_to(ROOT)) for d in DOCS])
def test_named_paths_exist(doc):
    missing = [n for n in _named_paths(doc) if not _exists(n, doc)]
    assert not missing, f"{doc.relative_to(ROOT)} names paths that do not exist: {missing}"


def test_claude_md_imports_exist():
    imports = re.findall(r"^@(\S+)", (ROOT / "CLAUDE.md").read_text(encoding="utf-8"), flags=re.M)
    assert imports and all((ROOT / i).exists() for i in imports)


def _records() -> dict[str, tuple[str, str]]:
    """number -> (title, status) from each record file."""
    out = {}
    for f in sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")):
        if f.name.startswith("0000"):
            continue
        text = f.read_text(encoding="utf-8")
        number, title = re.match(r"# (\d{4})\. (.+)", text).groups()
        assert f.name.startswith(number), f"{f.name}: heading number {number}"
        status = re.search(r"^- Status: (.+)$", text, flags=re.M).group(1).strip()
        out[number] = (title.strip(), status)
    return out


def test_decision_records_are_consistent():
    records = _records()
    index = (DECISIONS / "README.md").read_text(encoding="utf-8")
    rows = {m[0]: (m[1].strip(), m[2].strip())
            for m in re.findall(r"^\| \[(\d{4})\]\([^)]+\) \| (.+?) \| (.+?) \|$", index, flags=re.M)}
    assert rows.keys() == records.keys(), "index and record files differ"
    for number, (title, status) in records.items():
        assert rows[number] == (title, status), f"{number}: index says {rows[number]}, record says {(title, status)}"
        m = STATUS.match(status)
        assert m, f"{number}: unknown status {status!r}"
        if m.group(1):
            assert m.group(1) in records, f"{number} is superseded by a record that does not exist"
