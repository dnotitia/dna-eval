"""``scripts/license_report.py`` finds each task's data without downloading anything."""
import importlib.util
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "license_report.py"


def _module():
    spec = importlib.util.spec_from_file_location("license_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look their module up while it loads
    spec.loader.exec_module(module)          # defines functions only; stubs go in when discover() runs
    return module


def test_every_task_is_found_with_its_pinned_data():
    out = subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True, text=True,
                         check=True, cwd=ROOT)
    found = {t["task"]: t for t in json.loads(out.stdout)}
    assert {"performance", "style", "gpqa_diamond", "hle", "kmmlu_pro", "aime2025", "ifbench"} <= set(found)
    assert not any(t["error"] for t in found.values())
    # every task finds its data, unless it needs an optional package this environment lacks;
    # an agent task may also ship its data inside its package (no source to report)
    assert all(t["sources"] or t["missing"] or ".agentic." in t["module"] for t in found.values())
    assert all(t["sources"] for t in found.values() if ".agentic." not in t["module"])
    assert [s["name"] for s in found["hle"]["sources"]] == ["cais/hle"]
    assert found["gpqa_diamond"]["sources"][0]["kind"] == "url"
    assert all(len(s["revision"] or "") == 40 for t in found.values() for s in t["sources"] if s["kind"] == "hub")


def test_no_connection_is_made_while_tasks_are_built():
    m = _module()
    with m._no_network():
        try:
            socket.create_connection(("127.0.0.1", 9))
        except m.NetworkBlocked:
            pass
        else:
            raise AssertionError("a connection was allowed")
    assert socket.socket.connect.__name__ != "refuse"          # restored afterwards


def test_a_task_with_no_data_found_is_noted():
    m = _module()
    text = m.report([m.TaskData("odd_task", "dna_eval.tasks.benchmarks.odd_task", [])], offline=True)
    assert "| `odd_task` | — | — |  | no data found |" in text
