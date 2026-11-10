"""scripts/repeat_stddev.py: scores of repeated evaluations, their sample standard deviation."""
import itertools
import statistics
import subprocess
import sys
from pathlib import Path

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import match

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repeat_stddev.py"


def evaluate(log_dir, replies):
    """A four-question task under the name ``demo``, the model answering ``replies`` in turn."""
    model = get_model("mockllm/model", custom_outputs=itertools.cycle(
        [ModelOutput.from_content("mockllm/model", r) for r in replies]))
    task = Task(dataset=MemoryDataset([Sample(input=f"q{i}", target="A") for i in range(4)]),
                scorer=match(), name="demo")
    inspect_eval(task, model=model, display="none", log_dir=str(log_dir), max_connections=1)


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_sample_standard_deviation_of_the_scores(tmp_path):
    dirs = [tmp_path / f"run-{i}" for i in range(1, 4)]
    for d, replies in zip(dirs, (["A", "A", "A", "A"], ["A", "A", "B", "B"], ["A", "B", "B", "B"])):
        evaluate(d, replies)                                         # accuracy 100, 50, 25
    out = run(*dirs)
    assert out.returncode == 0, out.stderr
    line = next(l for l in out.stdout.splitlines() if l.startswith("demo"))
    assert "accuracy" in line and "100.00" in line and "50.00" in line and "25.00" in line
    sd = statistics.stdev([100, 50, 25])                             # n-1
    assert f"{sd:.3f}" in line and f"largest sd: {sd:.3f}" in out.stdout


def test_missing_task_is_reported(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    evaluate(a, ["A"])
    b.mkdir()
    out = run(a, b)
    assert out.returncode == 1 and "missing in" in out.stdout
