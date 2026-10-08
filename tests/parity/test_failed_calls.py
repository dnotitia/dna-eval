"""A judge call that fails is scored 0 and still counts in the averages.

A failed or unreadable judge call is retried twice (``rubric_scorer(judge_retries=2)``);
a sample still failing is scored 0 and stays in the average, as DeepEval's evaluation with
``ignore_errors=True`` and a re-judge loop treats it. Here the judge (``fake_openai.FakeOpenAI``) answers HTTP 500 or an unreadable reply when its
prompt carries the marker, which ``target`` puts there via ``expected_output``.
"""
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample

from dna_eval.generation import dna_generate_config
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate

CASES = {"judge_ok": (0.8, None), "judge_unparsable": (0.0, "parse"), "judge_http_500": (0.0, "call")}


@pytest.fixture(scope="module")
def run(fake_openai, models, tmp_path_factory):
    srv, (subject, judge) = fake_openai, models
    task = Task(dataset=[Sample(id=n, input="CASE:plain", target=f"JUDGE:{n};") for n in CASES],
                solver=dna_generate(), scorer=rubric_scorer("performance"),
                config=dna_generate_config(True))
    log = inspect_eval(task, model=subject, model_roles={"grader": judge},
                       max_retries=1, display="none", log_dir=str(tmp_path_factory.mktemp("logs")))[0]
    yield log, srv


@pytest.mark.parametrize("case", list(CASES))
def test_failed_judge_scores_zero(run, case):
    log, _ = run
    value, error = CASES[case]
    score = next(iter(next(s for s in log.samples if s.id == case).scores.values()))
    assert score.value == pytest.approx(value)
    assert score.metadata.get("judge_error") == error


def test_failed_judge_counts_in_average(run):
    log, _ = run
    assert log.status == "success"
    metrics = {k: v.value for s in log.results.scores for k, v in s.metrics.items()}
    assert metrics["mean"] == pytest.approx(0.8 / 3)
    assert metrics["trimmed_mean"] == pytest.approx(0.8 / 3)


def test_unreadable_reply_is_rejudged_twice(run):
    _, srv = run
    assert sum("JUDGE:judge_unparsable;" in p for p in srv.judge_prompts()) == 3
