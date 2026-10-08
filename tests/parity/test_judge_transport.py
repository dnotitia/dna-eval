"""A judge reply read over HTTP gives the score DeepEval's GEval gives it.

``test_judge_reply.py`` checks ``read_verdict`` on reply text. This test sends the same replies, and whole message
shapes such as ``reasoning_content`` from a judge served with a reasoning parser, through the
real path: the judge (``fake_openai.FakeOpenAI``) answers, Inspect's provider turns the
response into ``completion``, and ``rubric_scorer`` reads it. Expected values come from
``fixtures/judge_outputs.json``; a reply DeepEval fails on must end as a 0 after retries.
"""
import json
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample

from dna_eval.generation import dna_generate_config
from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import dna_generate

FIXTURE = json.loads((Path(__file__).parent.parent / "fixtures" / "judge_outputs.json").read_text(encoding="utf-8"))
REPLIES = {f"text_{c['id']}": ({"content": c["raw"]}, c["expected"]) for c in FIXTURE["cases"]}
REPLIES |= {f"msg_{c['id']}": (c["message"], c["expected"]) for c in FIXTURE["transport"]}


@pytest.fixture(scope="module")
def scores(fake_openai, models, tmp_path_factory):
    subject, judge = models
    fake_openai.judge_replies.update({k: msg for k, (msg, _) in REPLIES.items()})
    task = Task(dataset=[Sample(id=k, input="CASE:plain", target=f"JUDGE:{k};") for k in REPLIES],
                solver=dna_generate(), scorer=rubric_scorer("performance"),
                config=dna_generate_config(True))
    log = inspect_eval(task, model=subject, model_roles={"grader": judge},
                       max_retries=1, display="none", log_dir=str(tmp_path_factory.mktemp("logs")))[0]
    assert log.status == "success"
    return {s.id: next(iter(s.scores.values())) for s in log.samples}


@pytest.mark.parametrize("case", list(REPLIES))
def test_judge_reply_over_http_matches_deepeval(scores, case):
    score, expected = scores[case], REPLIES[case][1]
    if "error" in expected:
        assert score.value == 0.0 and score.metadata.get("judge_error")
    else:
        assert not score.metadata.get("judge_error")
        assert (score.value, score.explanation) == (pytest.approx(expected["score"]), expected["reason"])
