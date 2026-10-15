"""The documents people and agents follow stay true to the repository.

- Every repository path a document names (in backticks or as a relative link) exists, except
  the docs' hypothetical example task (names starting with ``my_``).
- CLAUDE.md imports only files that exist.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "dna_eval"
DOCS = sorted([ROOT / n for n in ("README.md", "AGENTS.md", "CLAUDE.md")]
              + list((ROOT / "docs").rglob("*.md")))
ROOT_FILES = {"README.md", "AGENTS.md", "CLAUDE.md", "pyproject.toml", "LICENSE", "NOTICE"}
PATHLIKE = re.compile(r"^[\w.\-/]+(\.(py|md|yaml|jinja|json|txt|toml)|/)$")


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
