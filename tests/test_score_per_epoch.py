"""scripts/score_per_epoch.py: one score.log line per run (epoch) and score.

- A DNA suite log is labelled per rubric and counts ``passed`` from the rubric threshold.
- A Korean benchmark log is labelled with its task's name and counts the full scores as passed.
- Every scorer a benchmark task uses is one the script records, so a new one cannot be dropped
  silently.
"""
import ast
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from dna_eval.scorers.answer_judge import answer_judge
from dna_eval.scorers.checklist_grade import checklist_grade
from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.scorers.math_answer import math_answer
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.scorers.simpleqa_grade import simpleqa_grade
from dna_eval.tasks.benchmarks import kbl, kmmlu_pro
from scripts.score_per_epoch import BENCHMARK_SCORERS, LABELS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_per_epoch.py"
BENCHMARKS = ROOT / "src" / "dna_eval" / "tasks" / "benchmarks"


def mock(*replies):
    return get_model("mockllm/model", custom_outputs=itertools.cycle(
        [ModelOutput.from_content("mockllm/model", r) for r in replies]))


def evaluate(task, log_dir, replies, grades=("unused",)):
    log = inspect_eval(task, model=mock(*replies), model_roles={"grader": mock(*grades)}, display="none",
                       log_dir=str(log_dir), max_connections=1)[0]
    assert log.status == "success", log.error
    return log


def record(tmp_path, *logs):
    """Run the script on ``logs``; returns (process, score.log lines, score.python.list.log rows)."""
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(SCRIPT), *map(str, logs), "--out-dir", str(out)],
                          capture_output=True, text=True)
    read = lambda name: (out / name).read_text().splitlines() if (out / name).exists() else []  # noqa: E731
    rows = [json.loads(line.rstrip(" ,")) for line in read("score.python.list.log")]
    assert all(re.fullmatch(r"\d{14}", row[0]) for row in rows)
    return proc, read("score.log"), [row[1:] for row in rows]


# --- Korean benchmarks ----------------------------------------------------------------------

# scorer -> (scorer, target, the model's replies, the grader's replies, score sum, passed) of 4 questions
CASES = {
    "letter_choice": (lambda: letter_choice([kmmlu_pro.ANSWER_PATTERN], letters="ABCDE"), "A",
                      ["정답: A", "정답: B"], ["unused"], 2.0, 2),
    "math_answer": (lambda: math_answer("integer"), "42", ["\\boxed{42}", "\\boxed{7}"], ["unused"], 2.0, 2),
    "kbl_answer": (kbl.kbl_answer, "A", ["답변: A", "답변: B"], ["unused"], 2.0, 2),
    "simpleqa_grade": (simpleqa_grade, "우주", ["우주입니다."], ["A", "B", "C", "A"], 2.0, 2),
    "answer_judge": (answer_judge, "1", ["\\boxed{1}"], ["true", "false"], 2.0, 2),
    # partial credit: the mean takes it, passed counts only the answers that met every item
    "checklist_grade": (checklist_grade, "", ["답입니다."],
                        [f"<score>\n{k}/2\n</score>" for k in (2, 1, 0, 1)], 2.0, 1),
}


def test_cases_cover_every_benchmark_scorer():
    assert set(CASES) == set(BENCHMARK_SCORERS)


@pytest.mark.parametrize("name", CASES)
def test_benchmark_scorer_is_recorded_under_the_task_name(tmp_path, name):
    scorer, target, replies, grades, total, passed = CASES[name]
    metadata = {"letters": "ABCDE", "question": "질문", "checklist": ["하나", "둘"]}
    task = Task(dataset=MemoryDataset([Sample(input=f"q{i}", target=target, metadata=metadata) for i in range(4)]),
                solver=generate(), scorer=scorer(), name="demo_bench")
    log = evaluate(task, tmp_path / "logs", replies, grades)
    assert list(log.samples[0].scores) == [name]
    proc, score_log, rows = record(tmp_path, tmp_path / "logs")
    assert proc.returncode == 0, proc.stderr
    assert score_log == [f'planets.loc[len(planets)] = ["model (demo_bench)", {total / 4:.2f}, {passed}]']
    assert rows == [["model", "demo_bench", total, passed, 4, total / 4]]


def test_benchmark_task_gets_one_line_per_run(tmp_path, monkeypatch):
    rows = [{"question": "1+1은?", "options": ["1", "2", "3", "4"], "solution": 2, "score": 3.0,
             "license_name": "관세사", "subject": "회계학"},
            {"question": "2+2는?", "options": ["1", "2", "3", "4"], "solution": 4, "score": 1.0,
             "license_name": "관세사", "subject": "무역영어"},
            {"question": "3+3은?", "options": ["6", "7", "8", "9"], "solution": 1, "score": 2.0,
             "license_name": "약사", "subject": "생명약학"}]
    path = tmp_path / "kmmlu_pro.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(kmmlu_pro, "hf_hub_download", lambda *a, **k: str(path))
    evaluate(kmmlu_pro.kmmlu_pro(epochs=2), tmp_path / "logs", ["정답: B"])       # right for the first question only
    proc, score_log, list_rows = record(tmp_path, tmp_path / "logs")
    assert proc.returncode == 0, proc.stderr
    assert score_log == ['planets.loc[len(planets)] = ["model (kmmlu_pro)", 0.33, 1]'] * 2
    assert list_rows == [["model", "kmmlu_pro", 1.0, 1, 3, 0.33333]] * 2


def test_every_benchmark_task_scorer_is_recorded():
    """The scorer each benchmark task passes to ``Task(scorer=...)`` is in ``BENCHMARK_SCORERS``."""
    used = {}
    for module in sorted(BENCHMARKS.glob("*.py")):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Task":
                call = next(k.value for k in node.keywords if k.arg == "scorer")
                assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name), module.name
                used[module.stem] = call.func.id
    registry = (ROOT / "src" / "dna_eval" / "_registry.py").read_text(encoding="utf-8")
    assert set(used) == set(re.findall(r"^from dna_eval\.tasks\.benchmarks\.(\w+) import", registry, flags=re.M))
    missing = {task: scorer for task, scorer in used.items() if scorer not in BENCHMARK_SCORERS}
    assert not missing, f"score_per_epoch.py does not record these benchmark scorers: {missing}"
    assert set(used.values()) == set(BENCHMARK_SCORERS) and not set(BENCHMARK_SCORERS) & set(LABELS)


def test_unknown_scorer_writes_nothing(tmp_path):
    task = Task(dataset=MemoryDataset([Sample(input="q", target="A")]), scorer=match(), name="other")
    evaluate(task, tmp_path / "logs", ["A"])
    proc, score_log, rows = record(tmp_path, tmp_path / "logs")
    assert proc.returncode == 1 and "nothing written" in proc.stderr
    assert score_log == [] and rows == []


# --- DNA suite ------------------------------------------------------------------------------

def test_suite_lines_keep_their_labels_and_threshold_passes(tmp_path):
    """Suite scorers are unchanged: fixed labels, per-category lines, ``passed`` from the rubric
    threshold (0.5 is a score, not a pass)."""
    samples = [Sample(input=f"q{i}", metadata={"category": c}) for i, c in enumerate(["grounding", "unknowable"] * 2)]
    for directory, raw in (("high", 10), ("low", 5)):
        task = Task(dataset=MemoryDataset(samples), solver=generate(), name="hallucination",
                    scorer=[rubric_scorer("hallucination"), rubric_scorer("repetition")])
        evaluate(task, tmp_path / directory, ["답변입니다."], [f'{{"score": {raw}, "reason": "fixture"}}'])
    proc, score_log, rows = record(tmp_path, tmp_path / "high", tmp_path / "low")
    assert proc.returncode == 0, proc.stderr
    assert score_log == [
        'planets.loc[len(planets)] = ["model (Hallu)", 1.00, 4]',
        'planets.loc[len(planets)] = ["model (Hallu:grounding)", 1.00, 2]',
        'planets.loc[len(planets)] = ["model (Hallu:unknowable)", 1.00, 2]',
        'planets.loc[len(planets)] = ["model (Rep)", 1.00, 4]',
        'planets.loc[len(planets)] = ["model (Hallu)", 0.50, 0]',
        'planets.loc[len(planets)] = ["model (Hallu:grounding)", 0.50, 0]',
        'planets.loc[len(planets)] = ["model (Hallu:unknowable)", 0.50, 0]',
        'planets.loc[len(planets)] = ["model (Rep)", 0.50, 0]',
    ]
    assert rows[0] == ["model", "Hallucination", 4.0, 4, 4, 1.0]
    assert rows[5] == ["model", "Hallucination:grounding", 1.0, 0, 2, 0.5]
