"""scripts/repeat_stddev.py: scores of repeated evaluations, their sample standard deviation."""
import itertools
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalMetric, EvalResults, EvalScore, HeadlineMetric
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.tasks import HEADLINE
from scripts.repeat_stddev import score

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repeat_stddev.py"


def evaluate(log_dir, replies):
    """A four-question task under the name ``demo``, the model answering ``replies`` in turn."""
    model = get_model("mockllm/model", custom_outputs=itertools.cycle(
        [ModelOutput.from_content("mockllm/model", r) for r in replies]))
    task = Task(dataset=MemoryDataset([Sample(input=f"q{i}", target="A") for i in range(4)]),
                scorer=match(), name="demo")
    return inspect_eval(task, model=model, display="none", log_dir=str(log_dir), max_connections=1)[0]


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


def evaluate_multiple_scorers(log_dir, raw_scores):
    """Two shipped rubric scorers with deliberately different trends across runs."""
    task = Task(dataset=[Sample(input="물의 화학식은?")], solver=generate(),
                scorer=[rubric_scorer("language_confusion"), rubric_scorer("repetition")],
                headline_metric=HEADLINE, name="multi")
    subject = get_model("mockllm/subject", custom_outputs=itertools.cycle(
        [ModelOutput.from_content("mockllm/subject", "H₂O입니다.")]))
    grader = get_model("mockllm/grader", custom_outputs=itertools.cycle(
        [ModelOutput.from_content("mockllm/grader", f'{{"score": {value}, "reason": "fixture"}}')
         for value in raw_scores]))
    log = inspect_eval(task, model=subject, model_roles={"grader": grader},
                       display="none", log_dir=str(log_dir), max_connections=1)[0]
    assert log.status == "success", log.error
    return log


def test_shared_metric_names_do_not_hide_variation(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    for directory, raw_scores in zip(dirs, ([10, 5], [5, 5])):
        evaluate_multiple_scorers(directory, raw_scores)
    out = run(*dirs)
    assert out.returncode == 0, out.stderr
    line = next(line for line in out.stdout.splitlines() if line.startswith("multi"))
    assert line.split()[1:4] == ["trimmed_mean", "100.00", "50.00"]
    assert f"largest sd: {statistics.stdev([100, 50]):.3f}" in out.stdout


@pytest.fixture(scope="module")
def base_log(tmp_path_factory):
    return evaluate(tmp_path_factory.mktemp("repeat-log"), ["A"])


def result(scorer, value, *, name="answer", reducer=None, **metrics):
    return EvalScore(name=name, scorer=scorer, reducer=reducer,
                     metrics={key: EvalMetric(name=key, value=number)
                              for key, number in {"trimmed_mean": value, **metrics}.items()})


def with_results(base_log, scores, headline=None):
    log = base_log.model_copy(deep=True)
    log.eval.headline_metric = None
    log.results = EvalResults(scores=scores, headline=headline)
    return log


@pytest.mark.parametrize("reverse", [False, True])
def test_headline_keeps_scorer_score_and_reducer_identity(base_log, reverse):
    scores = [result("primary", 0.9, name="reasoning", reducer="mean"),
              result("primary", 0.8, name="answer", reducer="mean"),
              result("primary", 0.25, name="answer", reducer="max"),
              result("other", 0.5, name="answer", reducer="max")]
    if reverse:
        scores.reverse()
    log = with_results(base_log, scores, HeadlineMetric(
        scorer="primary", score="answer", reducer="max", metric="trimmed_mean"))
    assert score(log, None) == ("trimmed_mean", 25.0)


def test_metric_override_stays_in_selected_score(base_log):
    log = with_results(base_log, [result("primary", 1, mean=0.75), result("other", 0.5, mean=0.25)],
                       HeadlineMetric(scorer="primary", metric="trimmed_mean"))
    assert score(log, "mean") == ("mean", 75.0)


def test_metric_override_cannot_select_another_scorer(base_log):
    log = with_results(base_log, [result("primary", 1), result("other", 0.5, other_only=0.25)],
                       HeadlineMetric(scorer="primary", metric="trimmed_mean"))
    with pytest.raises(ValueError, match="primary.*other_only"):
        score(log, "other_only")


def test_empty_metric_name_is_an_explicit_override(base_log):
    selected = result("primary", 1)
    selected.metrics[""] = EvalMetric(name="", value=0.25)
    log = with_results(base_log, [selected], HeadlineMetric(metric="trimmed_mean"))
    assert score(log, "") == ("", 25.0)


def test_legacy_log_without_headline_uses_first_score_first_metric(base_log):
    log = with_results(base_log, [EvalScore(name="match", scorer="match", metrics={
        "accuracy": EvalMetric(name="accuracy", value=0.75)})])
    assert score(log, None) == ("accuracy", 75.0)


def test_older_log_uses_task_headline_declaration(base_log):
    log = with_results(base_log, [result("primary", 1), result("other", 0.5)])
    log.eval.headline_metric = HeadlineMetric(scorer="primary", metric="trimmed_mean")
    assert score(log, None) == ("trimmed_mean", 100.0)


@pytest.mark.parametrize("results", [None, EvalResults(), EvalResults(scores=[
    EvalScore(name="empty", scorer="empty")])])
def test_missing_metrics_fail_clearly(base_log, results):
    log = base_log.model_copy(deep=True)
    log.results = results
    with pytest.raises(ValueError, match="demo: no headline metric"):
        score(log, None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e308])
def test_nonfinite_percentage_fails_clearly(base_log, value):
    log = with_results(base_log, [result("primary", value)], HeadlineMetric(metric="trimmed_mean"))
    with pytest.raises(ValueError, match="non-finite"):
        score(log, None)


def test_cli_reports_invalid_metric_without_traceback(tmp_path):
    dirs = [tmp_path / "a", tmp_path / "b"]
    for directory in dirs:
        evaluate(directory, ["A"])
    out = run(*dirs, "--metric", "missing")
    assert out.returncode == 2
    assert "missing" in out.stderr and "error:" in out.stderr
    assert "Traceback" not in out.stderr


def test_metric_override_finds_the_scorer_s_other_reducer(base_log):
    """A scorer reports ``mean`` under its reducer and ``trimmed_mean`` without one (rubric scorers)."""
    log = with_results(base_log, [result("primary", 0.2, name="primary", reducer="mean", mean=0.75),
                                  result("primary", 1, name="primary"), result("other", 0.5, mean=0.25)],
                       HeadlineMetric(scorer="primary", score="primary", metric="trimmed_mean"))
    log.results.scores[0].metrics.pop("trimmed_mean")
    assert score(log, "mean") == ("mean", 75.0)
    assert score(log, None) == ("trimmed_mean", 100.0)
