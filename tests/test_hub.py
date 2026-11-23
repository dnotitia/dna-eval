"""dna_eval.utils.hub: a config's split file found through the dataset card, read as CSV or JSON."""
import json

import pytest

from dna_eval.utils import hub

CARD = """---
configs:
- config_name: tables
  data_files:
  - split: dev
    path: data/tables-dev.csv
  - split: test
    path: data/tables-test.csv
- config_name: records
  data_files:
  - split: test
    path: records/v1.json
- config_name: other
  data_files:
  - split: test
    path: other.parquet
---
"""
FILES = {"README.md": CARD, "data/tables-dev.csv": "question,answer\nq0,1\n",
         "data/tables-test.csv": 'question,answer\n"a, b",2\n', "records/v1.json": json.dumps([{"q": "x", "n": 3}]),
         "other.parquet": ""}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo whose files are served from ``tmp_path``; a new name per test, as the card is cached."""
    for name, text in FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(hub, "hf_hub_download", lambda repo, filename, **kw: str(tmp_path / filename))
    return f"org/{tmp_path.name}"


def test_config_files_follow_the_card(repo):
    assert hub.config_files(repo, "rev") == {("tables", "dev"): "data/tables-dev.csv",
                                             ("tables", "test"): "data/tables-test.csv",
                                             ("records", "test"): "records/v1.json",
                                             ("other", "test"): "other.parquet"}


def test_read_config_reads_csv_as_strings_and_json_as_is(repo):
    assert hub.read_config(repo, "tables", "dev", "rev") == [{"question": "q0", "answer": "1"}]
    assert hub.read_config(repo, "tables", "test", "rev") == [{"question": "a, b", "answer": "2"}]
    assert hub.read_config(repo, "records", "test", "rev") == [{"q": "x", "n": 3}]


def test_read_config_rejects_other_formats_and_unknown_configs(repo):
    with pytest.raises(ValueError, match="unsupported data file"):
        hub.read_config(repo, "other", "test", "rev")
    with pytest.raises(KeyError):
        hub.read_config(repo, "tables", "train", "rev")
