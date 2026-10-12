"""The answer the judge sees, for every kind of reply and every kind of failed call to the
model under evaluation.

The model under evaluation is ``fake_openai.FakeOpenAI``, scripted to reply with surrounding
whitespace, separate reasoning, think tags, nothing, HTTP errors, too slowly, or to fail once
and then answer. For each reply the judge prompt must equal the prompt built from the
expected answer in ``fixtures/subject_outputs.json``, and every sample must be scored
rather than erroring out.
"""
import json
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample

from dna_eval.prompts import build_prompt
from dna_eval.rubrics import load_rubric
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import ATTEMPTS, REASONING_KEY, dna_generate
from dna_eval.tasks.dna_suite import dna_generate_config

FIXTURE = json.loads((Path(__file__).parent.parent / "fixtures" / "subject_outputs.json").read_text(encoding="utf-8"))
EXPECTED = FIXTURE["actual_output"]
EXPECTED_REASONING = FIXTURE["reasoning"]


@pytest.fixture(scope="module")
def run(fake_openai, models, tmp_path_factory):
    srv, (subject, judge) = fake_openai, models
    task = Task(dataset=[Sample(id=n, input=f"CASE:{n}", target="-") for n in EXPECTED],
                solver=dna_generate(), scorer=rubric_scorer("performance"),
                config=dna_generate_config(True))
    # attempt_timeout 1 s makes the `slow` reply time out; max_retries=1 only keeps
    # Inspect's own backoff short.
    log = inspect_eval(task, model=subject, model_roles={"grader": judge},
                       attempt_timeout=1, max_retries=1, display="none",
                       log_dir=str(tmp_path_factory.mktemp("logs")))[0]
    yield log, srv


def test_every_reply_is_scored(run):
    log, _ = run
    assert log.status == "success"
    assert all(s.error is None and s.scores for s in log.samples)


@pytest.mark.parametrize("case", list(EXPECTED))
def test_judge_sees_deepeval_answer(run, case):
    _, srv = run
    prompt = build_prompt(load_rubric("performance"), input_text=f"CASE:{case}",
                          actual_output=EXPECTED[case], expected_output="-")
    assert prompt in srv.judge_prompts()


def test_empty_reply_is_retried(run):
    _, srv = run
    assert len(srv.subject_requests("empty")) == ATTEMPTS


@pytest.mark.parametrize("case", ["http_500_then_answer", "empty_then_answer"])
def test_failed_call_recovers_on_retry(run, case):
    _, srv = run
    assert len(srv.subject_requests(case)) == 2


@pytest.mark.parametrize("case", list(EXPECTED_REASONING))
def test_reasoning_is_split_off_like_deepeval(run, case):
    """The reasoning a style judge sees: from a reasoning field or a leading <think> block,
    stripped, or a placeholder when there is none."""
    log, _ = run
    sample = next(s for s in log.samples if s.id == case)
    assert sample.store[REASONING_KEY] == EXPECTED_REASONING[case]
