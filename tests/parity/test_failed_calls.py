"""A judge call that fails is scored 0 and still counts in the averages.

A failed or unreadable judge call is retried twice (``rubric_scorer(judge_retries=2)``);
a sample still failing is scored 0 and stays in the average, as DeepEval's evaluation with
``ignore_errors=True`` and a re-judge loop treats it. Here the judge (``fake_openai.FakeOpenAI``)
answers HTTP 500 or an unreadable reply when its prompt carries the marker, which ``target``
puts there via ``expected_output``.

For a composite rubric DeepEval re-evaluates the whole test case, so when one component's
judge call fails every component is judged again (the last two tests, with ``mockllm``).
"""
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample

from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate
from dna_eval.tasks.dna_suite import dna_generate_config

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


def _composite_run(judge_replies: list[str], tmp_path):
    """One sample judged by the ``uncensorship`` composite; the judge answers ``judge_replies``
    in call order (bias, quality, bias, quality, ...), cycling."""
    import itertools

    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.solver import generate

    from dna_eval.scorers.rubric_scorer import rubric_scorer

    mk = lambda texts: get_model("mockllm/model", custom_outputs=(  # noqa: E731
        ModelOutput.from_content("mockllm/model", t) for t in itertools.cycle(texts)))
    task = Task(dataset=[Sample(input="q")], solver=generate(), scorer=rubric_scorer("uncensorship"))
    log = inspect_eval(task, model=mk(["답"]), model_roles={"grader": mk(judge_replies)},
                       display="none", log_dir=str(tmp_path))[0]
    sample = log.samples[0]
    calls = [e for e in sample.events if e.event == "model" and e.role == "grader"]
    return next(iter(sample.scores.values())), len(calls)


def test_composite_rejudges_every_component_when_one_fails(tmp_path):
    """Like DeepEval re-evaluating a failed test case: the component that succeeded is judged
    again too, and the second round's judgments are kept."""
    ok = lambda s: '{"score": %d, "reason": "ok"}' % s  # noqa: E731
    score, calls = _composite_run(["not json", ok(8), ok(9), ok(7)], tmp_path)
    assert calls == 4
    assert score.value == pytest.approx((0.9 + 0.7) / 2)
    assert {k: v["value"] for k, v in score.metadata["components"].items()} == {
        "uncensorship_bias": pytest.approx(0.9), "uncensorship_quality": pytest.approx(0.7)}


def test_composite_component_still_failing_scores_zero(tmp_path):
    score, calls = _composite_run(["not json", '{"score": 8, "reason": "ok"}'], tmp_path)
    assert calls == 6                                   # 3 rounds x 2 components
    comps = score.metadata["components"]
    assert comps["uncensorship_bias"]["value"] == 0.0 and comps["uncensorship_bias"]["judge_error"] == "parse"
    assert score.value == pytest.approx(0.8 / 2) and score.metadata["passed"] is False
