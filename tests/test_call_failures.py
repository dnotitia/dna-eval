"""``call_failures`` and ``judge_failures`` report how often a score stands for a failed call.

Failed calls keep their samples in every average (a placeholder answer is judged; a failed
judgment scores 0); these metrics make their share visible. Run against the scripted server in
``tests/parity/fake_openai.py``: the subject fails every attempt (HTTP 500), returns nothing,
returns only reasoning, or answers; one answered sample's judge fails every round.
"""
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import SampleScore, Score

from dna_eval.metrics.judge_failures import judge_failures
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import EMPTY_ANSWER, NO_ANSWER_KEY, dna_generate
from dna_eval.tasks import dna_generate_config
from tests.parity.fake_openai import FakeOpenAI

SAMPLES = {   # id: (subject case, judge marker, expected metadata["no_answer"])
    "ok": ("plain", "", None),
    "failed": ("http_500", "", "failed"),
    "empty": ("empty", "", "empty"),
    "reasoning_only": ("reasoning_only", "", "empty"),
    "judge_failed": ("plain", "JUDGE:judge_http_500;", None),
}


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    with FakeOpenAI() as srv:
        subject = get_model("vllm/fake-call-failures", base_url=srv.base_url, api_key="x")
        judge = get_model("openai-api/judge/fake-call-failures", base_url=srv.base_url, api_key="x")
        task = Task(dataset=[Sample(id=i, input=f"CASE:{c}", target=t) for i, (c, t, _) in SAMPLES.items()],
                    solver=dna_generate(), scorer=rubric_scorer("performance"),
                    config=dna_generate_config(True), epochs=2)
        yield inspect_eval(task, model=subject, model_roles={"grader": judge}, max_retries=1,
                           display="none", log_dir=str(tmp_path_factory.mktemp("logs")))[0]


@pytest.mark.parametrize("sid", list(SAMPLES))
def test_reason_for_no_answer_is_recorded(log, sid):
    expected = SAMPLES[sid][2]
    for sample in (s for s in log.samples if s.id == sid):
        assert sample.store.get(NO_ANSWER_KEY) == expected
        assert next(iter(sample.scores.values())).metadata.get("no_answer") == expected
        assert (sample.output.completion == EMPTY_ANSWER) == (expected is not None)


def test_rates_count_every_run(log):
    metrics = {k: v.value for s in log.results.scores for k, v in s.metrics.items()}
    assert metrics["answer_failed"] == pytest.approx(1 / 5)
    assert metrics["answer_empty"] == pytest.approx(2 / 5)
    assert metrics["judge_failed"] == pytest.approx(1 / 5)


def test_composite_counts_a_failed_component():
    scores = [SampleScore(score=Score(value=0.4, metadata={"components": {
                  "a": {"judge_error": None}, "b": {"judge_error": "parse"}}}), sample_id=1),
              SampleScore(score=Score(value=0.9, metadata={"components": {
                  "a": {"judge_error": None}, "b": {"judge_error": None}}}), sample_id=2)]
    assert judge_failures()(scores) == {"judge_failed": 0.5}
