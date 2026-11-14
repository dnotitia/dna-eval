"""Inspect limit errors propagate out of ``call_with_retries`` instead of being retried."""
import itertools

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model

from dna_eval.scorers.rubric_scorer import rubric_scorer
from dna_eval.solvers.dna_generate import EMPTY_ANSWER, EMPTY_REASONING, dna_generate


def _mock(text):
    return get_model("mockllm/model", custom_outputs=(ModelOutput.from_content("mockllm/model", text)
                                                      for _ in itertools.count()))


def test_token_limit_is_recorded_not_retried(tmp_path):
    task = Task(dataset=[Sample(id="s1", input="question")], solver=dna_generate(),
                scorer=rubric_scorer("performance"), token_limit=1)
    log = inspect_eval(task, model=_mock("a fairly long answer to the question"),
                       model_roles={"grader": _mock('{"score": 8, "reason": "ok"}')},
                       display="none", log_dir=str(tmp_path))[0]
    sample = log.samples[0]
    assert sample.limit is not None and sample.limit.type == "token"
    subject_calls = [e for e in sample.events if e.event == "model" and e.role != "grader"]
    assert len(subject_calls) == 1
    assert sample.store["reasoning"] == EMPTY_REASONING
    assert sample.output.completion == EMPTY_ANSWER
    judge_calls = [e for e in sample.events if e.event == "model" and e.role == "grader"]
    prompt = judge_calls[0].input[0].content   # long prompts are stored as attachment references
    prompt = sample.attachments.get(prompt.removeprefix("attachment://"), prompt)
    assert EMPTY_ANSWER in prompt
    for score in sample.scores.values():
        assert (score.metadata or {}).get("no_answer") != "failed"
