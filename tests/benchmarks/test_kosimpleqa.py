"""KoSimpleQA and the SimpleQA grader: grades, aggregate metrics, and a full run on mock models."""
import itertools

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, get_model

from dna_eval.scorers.simpleqa_grade import GRADER_TEMPLATE, read_grade
from dna_eval.tasks.benchmarks import kosimpleqa

RECORDS = [
    {"category": "art", "content": "질문 1", "answer": "우주"},
    {"category": "art", "content": "질문 2", "answer": "26번째"},
    {"category": "music", "content": "질문 3", "answer": "뚜두뚜두"},
    {"category": "music", "content": "질문 4", "answer": "1.7%"},
]


def mock(outputs):
    return get_model("mockllm/model", custom_outputs=itertools.cycle(outputs))


def test_read_grade_takes_the_first_letter_as_simple_evals_does():
    assert read_grade("A") == "A"
    assert read_grade("B\n") == "B"
    assert read_grade("The grade is C") == "C"        # "T", "h", "e" are not grades; the first A/B/C wins
    assert read_grade("Answer: B") == "A"              # simple-evals' regex picks the A of "Answer"
    assert read_grade("no grade") is None


def test_grader_template_is_the_simple_evals_text():
    assert GRADER_TEMPLATE.startswith("Your job is to look at a question, a gold target, and a predicted answer")
    assert GRADER_TEMPLATE.endswith('Just return the letters "A", "B", or "C", with no text around it.')
    assert GRADER_TEMPLATE.count("{question}") == GRADER_TEMPLATE.count("{target}") == 1
    assert 'gold target "120k". \n' in GRADER_TEMPLATE                   # trailing spaces as published


def test_kosimpleqa_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(kosimpleqa, "json_dataset", lambda *a, sample_fields, **k:
                        MemoryDataset([sample_fields(r) for r in RECORDS], name="kosimpleqa"))
    answers = [ModelOutput.from_content("mockllm/model", "우주입니다."),
               ModelOutput.from_content("mockllm/model", "25번째입니다."),
               ModelOutput.from_content("mockllm/model", "뚜두뚜두", stop_reason="max_tokens"),   # truncated
               ModelOutput.from_content("mockllm/model", "모르겠습니다.")]
    grades = [ModelOutput.from_content("mockllm/model", g) for g in ("A", "B", "C")]     # the 3rd answer is never graded
    log = inspect_eval(kosimpleqa.kosimpleqa(), model=mock(answers), model_roles={"grader": mock(grades)},
                       display="none", log_dir=str(tmp_path / "logs"))[0]
    assert log.status == "success", log.error
    m = {name: v.value for s in log.results.scores for name, v in s.metrics.items()}
    assert m["accuracy"] == pytest.approx(1 / 4)
    assert m["correct"] == pytest.approx(1 / 4) and m["incorrect"] == pytest.approx(1 / 4)
    assert m["not_attempted"] == pytest.approx(2 / 4)
    assert m["correct_given_attempted"] == pytest.approx(1 / 2)
    assert m["f_score"] == pytest.approx(2 * 0.5 * 0.25 / 0.75)
    assert m["truncated"] == pytest.approx(1 / 4) and m["judge_failed"] == 0.0
    assert m["category/art"] == pytest.approx(1 / 2) and m["category/music"] == 0.0
    by_id = {s.id: s for s in log.samples}
    truncated = by_id[3]
    assert truncated.scores["simpleqa_grade"].metadata["grade"] == "NOT_ATTEMPTED"
    assert not any(e.event == "model" and e.role == "grader" for e in truncated.events)
    graded = next(e for e in by_id[1].events if e.event == "model" and e.role == "grader")
    assert graded.input[0].content == "You are a helpful assistant." and graded.config.temperature == 0.5
    prompt = graded.input[1].text
    prompt = by_id[1].attachments.get(prompt.removeprefix("attachment://"), prompt)   # long texts are stored apart
    assert "Question: 질문 1\nGold target: 우주\nPredicted answer: 우주입니다." in prompt


def test_unreadable_grade_is_not_attempted_and_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(kosimpleqa, "json_dataset", lambda *a, sample_fields, **k:
                        MemoryDataset([sample_fields(RECORDS[0])], name="kosimpleqa"))
    log = inspect_eval(kosimpleqa.kosimpleqa(), model=mock([ModelOutput.from_content("mockllm/model", "우주")]),
                       model_roles={"grader": mock([ModelOutput.from_content("mockllm/model", "xyz")])},
                       display="none", log_dir=str(tmp_path / "logs"))[0]
    md = log.samples[0].scores["simpleqa_grade"].metadata
    assert md["grade"] == "NOT_ATTEMPTED" and md["judge_error"] == "parse"
    assert log.results.scores[0].metrics["accuracy"].value == 0.0


@pytest.mark.network
def test_kosimpleqa_data():
    try:
        samples = list(kosimpleqa.json_dataset(
            f"{kosimpleqa.REPO_URL}/{kosimpleqa.DATASET_REVISION}/{kosimpleqa.DATA_FILE}",
            sample_fields=kosimpleqa.record_to_sample, auto_id=True))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dataset not reachable: {exc}")
    assert len(samples) == 1000 and len({s.input for s in samples}) == 1000
    assert len({s.metadata["category"] for s in samples}) == 10
    assert all(s.target and s.metadata["question"] == s.input for s in samples)
