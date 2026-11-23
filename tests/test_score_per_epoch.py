"""scripts/score_per_epoch.py reports DNA suite logs only.

A log of another task is skipped when found in a directory, and is an error when given as a file
or named with ``--task``.
"""
import subprocess
import sys
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from dna_eval.scorers.rubric_scorer import rubric_scorer

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_per_epoch.py"


def _mock(text):
    return get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", text)] * 8)


def _logs(log_dir):
    """A persona log (a suite scorer) and a kmmlu_pro log (another scorer), in ``log_dir``."""
    samples = [Sample(input="q", target="A"), Sample(input="q", target="B")]
    suite = Task(dataset=samples, solver=generate(), scorer=rubric_scorer("persona"), name="persona")
    other = Task(dataset=samples, solver=generate(), scorer=match(), name="kmmlu_pro")
    logs = inspect_eval([suite, other], model=_mock("A"), display="none", log_dir=str(log_dir),
                        model_roles={"grader": _mock('{"score": 9, "reason": "ok"}')})
    assert all(log.status == "success" for log in logs)
    return {log.eval.task: log.location for log in logs}


def _record(tmp_path, *args):
    out = tmp_path / "out"
    proc = subprocess.run([sys.executable, str(SCRIPT), *map(str, args), "--out-dir", str(out)],
                          capture_output=True, text=True)
    score_log = out / "score.log"
    return proc, score_log.read_text().splitlines() if score_log.exists() else []


def test_directory_skips_other_tasks(tmp_path):
    _logs(tmp_path / "logs")
    proc, lines = _record(tmp_path, tmp_path / "logs")
    assert proc.returncode == 0, proc.stderr
    assert lines == ['planets.loc[len(planets)] = ["model (Pers)", 0.90, 2]']
    assert "kmmlu_pro is not a DNA suite task" in proc.stderr


def test_other_task_given_as_a_file_is_an_error(tmp_path):
    other = _logs(tmp_path / "logs")["kmmlu_pro"]
    proc, lines = _record(tmp_path, other)
    assert proc.returncode == 2
    assert "only DNA suite logs are supported" in proc.stderr
    assert lines == []


def test_other_task_named_with_task_is_an_error(tmp_path):
    _logs(tmp_path / "logs")
    proc, lines = _record(tmp_path, tmp_path / "logs", "--task", "persona,kmmlu_pro")
    assert proc.returncode == 2
    assert lines == []
