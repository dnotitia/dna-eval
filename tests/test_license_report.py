"""``scripts/license_report.py`` finds each task's data without downloading it, and flags
benchmarks missing from the licence table."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "license_report.py"


def _module():
    spec = importlib.util.spec_from_file_location("license_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # dataclasses look their module up while it loads
    spec.loader.exec_module(module)        # defines functions only; stubs go in when discover() runs
    return module


def test_every_task_is_found_with_its_pinned_data():
    out = subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True, text=True,
                         check=True, cwd=ROOT)
    found = {t["task"]: t for t in json.loads(out.stdout)}
    assert {"performance", "style", "gpqa_diamond", "hle", "kmmlu_pro", "aime2025", "ifbench"} <= set(found)
    assert all(t["sources"] and not t["error"] for t in found.values())
    assert found["hle"]["sources"] == [{"kind": "hub", "name": "cais/hle", "revision": found["hle"]["sources"][0]["revision"]}]
    assert found["gpqa_diamond"]["sources"][0]["kind"] == "url"
    assert all(len(s["revision"] or "") == 40 for t in found.values() for s in t["sources"] if s["kind"] == "hub")


def test_every_benchmark_is_in_the_licence_table():
    m = _module()
    table = m.table_licences()
    out = subprocess.run([sys.executable, str(SCRIPT), "--json"], capture_output=True, text=True,
                         check=True, cwd=ROOT)
    benchmarks = {t["task"] for t in json.loads(out.stdout) if ".benchmarks." in t["module"]}
    assert benchmarks <= set(table), f"add these tasks to docs/benchmarks.md: {sorted(benchmarks - set(table))}"


def test_a_benchmark_missing_from_the_table_needs_attention():
    m = _module()
    found = [m.TaskData("new_bench", "dna_eval.tasks.benchmarks.new_bench", [m.Source("hub", "org/data", "a" * 40)]),
             m.TaskData("performance", "dna_eval.tasks.dna_suite.performance", [m.Source("hub", "org/own", "b" * 40)])]
    text, attention = m.report(found, offline=True)
    assert attention == 1
    assert "| `new_bench` | org/data@aaaaaaa | — |  | not in the table |" in text
    assert "suite data" in text
