"""scripts/score_per_epoch.py: one line per run and score, for DNA suite and benchmark logs."""
import importlib
import json
import pkgutil
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import is_registry_object, registry_info
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate

import dna_eval.tasks.benchmarks as benchmarks
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.tasks import HEADLINE
from dna_eval.tasks.benchmarks import kmmlu_pro, kosimpleqa
from scripts.score_per_epoch import BENCHMARK_SCORERS, LABELS, timestamp

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_per_epoch.py"


def by_question(replies):
    """A mock model that answers the n-th time it sees a question with that question's n-th reply
    (``replies``: a text in the prompt -> one reply per epoch)."""
    seen = Counter()

    def output(messages, *_):
        key = next(k for k in replies if k in messages[-1].text)
        seen[key] += 1
        reply = replies[key][seen[key] - 1]
        return reply if isinstance(reply, ModelOutput) else ModelOutput.from_content("mockllm/model", reply)

    return get_model("mockllm/model", custom_outputs=output)


def evaluate(task, log_dir, model, **roles):
    """Samples one at a time, so each question's epochs reach the model in epoch order."""
    log = inspect_eval(task, model=model, model_roles=roles or None, display="none", log_dir=str(log_dir),
                       max_samples=1, max_connections=1)[0]
    assert log.status == "success", log.error
    return log


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def finished(log, epoch):
    """The list log's time for ``epoch``: when the run's last sample finished, in local time."""
    return timestamp(max(s.completed_at for s in log.samples if s.epoch == epoch))


def written(out):
    return (out / "score.log").read_text(), (out / "score.python.list.log").read_text()


# --- logs ---------------------------------------------------------------------------------------

KMMLU_ROWS = [      # answers B, E, A
    {"question": "1+1은?", "options": ["1", "2", "3", "4"], "solution": 2, "score": 3.0,
     "license_name": "관세사", "subject": "회계학"},
    {"question": "2+2는?", "options": ["1", "2", "3", "4", "5"], "solution": 5, "score": 1.0,
     "license_name": "관세사", "subject": "무역영어"},
    {"question": "3+3은?", "options": ["6", "7", "8", "9"], "solution": 1, "score": 2.0,
     "license_name": "약사", "subject": "생명약학"},
]

KOSIMPLEQA_RECORDS = [
    {"category": "art", "content": "질문 1", "answer": "우주"},
    {"category": "art", "content": "질문 2", "answer": "26번째"},
    {"category": "music", "content": "질문 3", "answer": "뚜두뚜두"},
    {"category": "music", "content": "질문 4", "answer": "1.7%"},
]
GRADES = {"우주입니다.": "A", "25번째입니다.": "B", "26번째입니다.": "A", "뚜두뚜두": "A",
          "모르겠습니다.": "C", "1.7%": "unreadable"}


def grader(messages, *_):
    """The SimpleQA grade of the predicted answer in the prompt."""
    predicted = re.search(r"^Predicted answer: (.*)$", messages[-1].text, re.M).group(1)
    return ModelOutput.from_content("mockllm/grader", GRADES[predicted])


def rubric_grader(messages, *_):
    """A rubric judgment that depends only on the answer being judged."""
    raw = next(score for marker, score in (("[good]", 10), ("[ok]", 7), ("[bad]", 2)) if marker in messages[-1].text)
    return ModelOutput.from_content("mockllm/grader", json.dumps({"score": raw, "reason": "fixture"}))


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    """One small log of each kind, all in one directory, each with two epochs."""
    log_dir = tmp_path_factory.mktemp("logs")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        rows = log_dir.parent / "kmmlu_pro.jsonl"
        rows.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in KMMLU_ROWS), encoding="utf-8")
        mp.setattr(kmmlu_pro, "hf_hub_download", lambda *a, **k: str(rows))
        # Run 1: all three right. Run 2: right, wrong letter, no answer.
        out["kmmlu_pro"] = evaluate(kmmlu_pro.kmmlu_pro(epochs=2), log_dir, by_question({
            "1+1은?": ["정답: B", "정답: B"], "2+2는?": ["정답: E", "정답: A"], "3+3은?": ["정답: A", "모르겠다"]}))

        mp.setattr(kosimpleqa, "json_dataset", lambda *a, sample_fields, **k:
                   MemoryDataset([sample_fields(r) for r in KOSIMPLEQA_RECORDS], name="kosimpleqa"))
        truncated = ModelOutput.from_content("mockllm/model", "뚜두뚜두", stop_reason="max_tokens")
        # Run 1: CORRECT, INCORRECT, truncated (not graded), NOT_ATTEMPTED.
        # Run 2: CORRECT, CORRECT, CORRECT, unreadable grade (a judge failure).
        out["kosimpleqa"] = evaluate(kosimpleqa.kosimpleqa(epochs=2, judge_retries=0), log_dir, by_question({
            "질문 1": ["우주입니다.", "우주입니다."], "질문 2": ["25번째입니다.", "26번째입니다."],
            "질문 3": [truncated, "뚜두뚜두"], "질문 4": ["모르겠습니다.", "1.7%"]}),
            grader=get_model("mockllm/grader", custom_outputs=grader))

    samples = [Sample(input=f"질문 {i}", target="전제가 틀렸다고 지적해야 한다", metadata={"category": c})
               for i, c in enumerate(["false_premise", "nonexistent", "false_premise"])]
    task = Task(dataset=samples, solver=generate(), scorer=rubric_scorer("hallucination"), epochs=2,
                headline_metric=HEADLINE, name="hallucination")
    out["hallucination"] = evaluate(task, log_dir, by_question({
        "질문 0": ["[good]", "[bad]"], "질문 1": ["[ok]", "[good]"], "질문 2": ["[bad]", "[ok]"]}),
        grader=get_model("mockllm/grader", custom_outputs=rubric_grader))
    return out


# --- output -------------------------------------------------------------------------------------

def test_letter_choice_benchmark_counts_correct_answers(logs, tmp_path):
    log = logs["kmmlu_pro"]
    assert {name for s in log.samples for name in s.scores} == {"letter_choice"}
    out = run(log.location, "--out-dir", tmp_path)
    assert out.returncode == 0, out.stderr
    assert written(tmp_path) == (
        'planets.loc[len(planets)] = ["model (kmmlu_pro)", 1.00, 3]\n'
        'planets.loc[len(planets)] = ["model (kmmlu_pro)", 0.33, 1]\n',
        f'["{finished(log, 1)}", "model", "kmmlu_pro", 3.0, 3, 3, 1.00000 ],\n'
        f'["{finished(log, 2)}", "model", "kmmlu_pro", 1.0, 1, 3, 0.33333 ],\n')
    # the per-run means are the task's plain accuracy, not its weighted headline
    metrics = {name: m.value for s in log.results.scores for name, m in s.metrics.items()}
    assert metrics["accuracy"] == pytest.approx((1 + 1 / 3) / 2)
    assert metrics["weighted_accuracy"] == pytest.approx((1 + 3 / 6) / 2)


def test_judge_graded_benchmark_counts_correct_grades_only(logs, tmp_path):
    log = logs["kosimpleqa"]
    failed = [s for s in log.samples if "judge_error" in s.scores["simpleqa_grade"].metadata]
    assert len(failed) == 1 and failed[0].epoch == 2
    out = run(log.location, "--out-dir", tmp_path, "--model-name", "my-model")
    assert out.returncode == 0, out.stderr
    assert written(tmp_path) == (
        'planets.loc[len(planets)] = ["my-model (kosimpleqa)", 0.25, 1]\n'
        'planets.loc[len(planets)] = ["my-model (kosimpleqa)", 0.75, 3]\n',
        f'["{finished(log, 1)}", "my-model", "kosimpleqa", 1.0, 1, 4, 0.25000 ],\n'
        f'["{finished(log, 2)}", "my-model", "kosimpleqa", 3.0, 3, 4, 0.75000 ],\n')


def test_dna_suite_lines_are_unchanged(logs, tmp_path):
    """The lines the script wrote for DNA suite logs before benchmarks were added, byte for byte."""
    log = logs["hallucination"]
    out = run(log.location, "--out-dir", tmp_path)
    assert out.returncode == 0, out.stderr
    assert written(tmp_path) == (
        'planets.loc[len(planets)] = ["model (Hallu)", 0.63, 2]\n'
        'planets.loc[len(planets)] = ["model (Hallu:false_premise)", 0.60, 1]\n'
        'planets.loc[len(planets)] = ["model (Hallu:nonexistent)", 0.70, 1]\n'
        'planets.loc[len(planets)] = ["model (Hallu)", 0.63, 2]\n'
        'planets.loc[len(planets)] = ["model (Hallu:false_premise)", 0.45, 1]\n'
        'planets.loc[len(planets)] = ["model (Hallu:nonexistent)", 1.00, 1]\n',
        f'["{finished(log, 1)}", "model", "Hallucination", 1.9, 2, 3, 0.63333 ],\n'
        f'["{finished(log, 1)}", "model", "Hallucination:false_premise", 1.2, 1, 2, 0.60000 ],\n'
        f'["{finished(log, 1)}", "model", "Hallucination:nonexistent", 0.7, 1, 1, 0.70000 ],\n'
        f'["{finished(log, 2)}", "model", "Hallucination", 1.9, 2, 3, 0.63333 ],\n'
        f'["{finished(log, 2)}", "model", "Hallucination:false_premise", 0.8999999999999999, 1, 2, 0.45000 ],\n'
        f'["{finished(log, 2)}", "model", "Hallucination:nonexistent", 1.0, 1, 1, 1.00000 ],\n')


@pytest.mark.parametrize("task", ["kmmlu_pro", "dna_eval/hallucination"])
def test_task_filter_keeps_only_that_task(logs, tmp_path, task):
    out = run(Path(logs["kmmlu_pro"].location).parent, "--out-dir", tmp_path, "--task", task)
    assert out.returncode == 0, out.stderr
    label = "(kmmlu_pro)" if task == "kmmlu_pro" else "(Hallu"
    score_log = written(tmp_path)[0].splitlines()
    assert len(score_log) == (2 if task == "kmmlu_pro" else 6) and all(label in line for line in score_log)


def test_directory_without_filter_writes_every_log(logs, tmp_path):
    out = run(Path(logs["kmmlu_pro"].location).parent, "--out-dir", tmp_path)
    assert out.returncode == 0, out.stderr
    assert len(written(tmp_path)[0].splitlines()) == 2 + 2 + 6


def test_unknown_scorer_writes_nothing(tmp_path):
    task = Task(dataset=[Sample(input="q", target="A")], scorer=match(), name="demo")
    log = evaluate(task, tmp_path / "logs", get_model("mockllm/model"))
    out = run(log.location, "--out-dir", tmp_path / "out")
    assert out.returncode == 1 and "nothing written" in out.stderr
    assert written(tmp_path / "out") == ("", "")


# --- coverage -----------------------------------------------------------------------------------

def benchmark_scorers():
    """The scorers each benchmark task module defines or imports, by registered name."""
    found = {}
    for module in pkgutil.iter_modules(benchmarks.__path__):
        mod = importlib.import_module(f"{benchmarks.__name__}.{module.name}")
        names = {registry_info(obj).name.split("/")[-1] for obj in vars(mod).values()
                 if is_registry_object(obj, type="scorer")}
        tasks = [obj for obj in vars(mod).values()
                 if is_registry_object(obj, type="task") and obj.__module__ == mod.__name__]
        if tasks:
            found[module.name] = names
    return found


def test_every_benchmark_scorer_has_a_label():
    found = benchmark_scorers()
    assert len(found) >= 12 and all(found.values()), found
    missing = {task: unknown for task, names in found.items() if (unknown := names - {*BENCHMARK_SCORERS, *LABELS})}
    assert not missing, f"benchmark scorers without a label (add them to BENCHMARK_SCORERS): {missing}"
