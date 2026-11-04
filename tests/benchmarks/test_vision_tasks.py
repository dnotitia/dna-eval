"""The multimodal benchmark tasks (HAERAE-VISION, KMMMU) and their judge scorers, on mock models."""
import base64
import csv
import itertools

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ContentImage, ModelOutput, get_model

from dna_eval.scorers.answer_judge import build_prompt as kmmmu_judge_prompt, read_verdict
from dna_eval.scorers.checklist_grade import build_prompt as checklist_prompt, prompt_template, read_score
from dna_eval.tasks.benchmarks import haerae_vision, kmmmu
from dna_eval.utils.images import image_data_uri, image_mime_type

# A 1x1 PNG.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


def mock(*texts, stop_reason="stop"):
    outs = [t if isinstance(t, ModelOutput) else ModelOutput.from_content("mockllm/model", t, stop_reason=stop_reason)
            for t in texts]
    return get_model("mockllm/model", custom_outputs=itertools.cycle(outs))


def run(task, tmp_path, model, grader):
    log = inspect_eval(task, model=model, model_roles={"grader": grader}, display="none",
                       log_dir=str(tmp_path / "logs"))[0]
    assert log.status == "success", log.error
    return log


def metrics(log):
    return {name: m.value for s in log.results.scores for name, m in s.metrics.items()}


def test_image_data_uri_keeps_the_bytes():
    assert image_mime_type(PNG) == "image/png"
    assert image_mime_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert image_mime_type(b"GIF89a....") == "image/gif"
    assert image_data_uri(PNG) == "data:image/png;base64," + base64.b64encode(PNG).decode()
    with pytest.raises(ValueError):
        image_mime_type(b"not an image")


# --- HAERAE-VISION ------------------------------------------------------------------------------

def test_checklist_score_reading():
    assert read_score("<evidence>...</evidence>\n<score>\n3/4\n</score>") == 0.75
    assert read_score("<score>0.5/2</score>") == 0.25
    assert read_score("<SCORE>1</SCORE>") == 1.0
    assert read_score("no tag") is None
    assert read_score("<score>1/2</score> <score>2/2</score>") is None       # more than one tag
    assert read_score("<score>1/0</score>") is None


def test_checklist_prompt_is_assembled_as_score_py_does():
    p = checklist_prompt("질문", "답변", ["1. 항목", "2. 항목"])
    assert p == prompt_template + "[INPUT]\n질문\n[Response]\n답변\n[Checklist]\n['1. 항목', '2. 항목']\n"
    assert "{{QUESTION}}" in prompt_template                                  # sent unfilled, as published


VISION_ROWS = [
    {"question_idx": 0, "source": "kin", "category": "게임", "images": [{"bytes": PNG, "path": None}] * 2,
     "question_original": "이거 뭐예요?", "question_explicit": "게임 X의 이 아이템은 무엇인가요?",
     "checklist": ["1. 아이템 이름을 말하나요?", "2. 용도를 설명하나요?"]},
    {"question_idx": 1, "source": "sci", "category": "과학", "images": [{"bytes": PNG, "path": None}],
     "question_original": "이 꽃 이름은?", "question_explicit": "사진 속 노란 꽃의 이름은 무엇인가요?",
     "checklist": ["1. 꽃 이름을 말하나요?"]},
]


@pytest.mark.parametrize("question_type", ["original", "explicit"])
def test_haerae_vision_samples(question_type):
    s = haerae_vision.record_to_sample(VISION_ROWS[0], question_type)
    assert s.id == 0 and s.input[0].role == "system" and s.input[0].content == haerae_vision.SYSTEM
    text, *images = s.input[1].content
    assert text.text == VISION_ROWS[0][f"question_{question_type}"] == s.metadata["question"]
    assert len(images) == 2 and all(isinstance(i, ContentImage) and i.image.startswith("data:image/png") for i in images)


def test_haerae_vision_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(haerae_vision, "load_records", lambda revision: VISION_ROWS)
    grader = mock("<evidence>...</evidence>\n<score>\n1/2\n</score>", "garbled", "garbled", "garbled")
    log = run(haerae_vision.haerae_vision(judge_retries=2), tmp_path, mock("아이템 X입니다."), grader)
    m = metrics(log)
    by_id = {s.id: s.scores["checklist_grade"] for s in log.samples}
    scores = sorted(s.value for s in by_id.values())
    assert scores == [0.0, 0.5]                                            # one judged 1/2, one never readable
    assert m["mean"] == pytest.approx(0.25) and m["judge_failed"] == pytest.approx(0.5)
    assert {s.metadata.get("judge_error") for s in by_id.values()} == {None, "parse"}
    assert log.results.headline.metric == "mean"


# --- KMMMU --------------------------------------------------------------------------------------

def test_kmmmu_judge_prompt_and_verdict():
    assert kmmmu_judge_prompt("Q", "4", "$\\boxed{4}$", "[4, '4']") == (
        "Question:\nQ\n\nGold answer:\n4\n\nModel output:\n$\\boxed{4}$\n\n"
        "Parsed model output (Math-Verify parse):\n[4, '4']\n\nIs the model answer correct? Reply only with true or false.")
    assert read_verdict(" True.") and read_verdict("true") and not read_verdict("false") and not read_verdict("It is true")


def kmmmu_csv(tmp_path):
    uri = image_data_uri(PNG)
    rows = [{"question": "그림의 값은?", "answer": "4", "question_type": "객관식(단일정답)", "image_link": repr([uri])},
            {"question": "옳은 것을 모두 고르면?", "answer": "ㄱ,ㄷ", "question_type": "객관식(복수정답)",
             "image_link": repr([uri, uri])}]
    path = tmp_path / "kmmmu.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_kmmmu_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(kmmmu, "hf_hub_download", lambda *a, **k: str(kmmmu_csv(tmp_path)))
    samples = kmmmu.load_samples(kmmmu.DATASET_REVISION)
    assert [s.target for s in samples] == ["4", "ㄱ,ㄷ"]
    assert samples[0].input[0].content == kmmmu.SYSTEM_PROMPT
    assert [type(c).__name__ for c in samples[1].input[1].content] == ["ContentText", "ContentImage", "ContentImage"]


def test_kmmmu_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("math_verify")
    monkeypatch.setattr(kmmmu, "hf_hub_download", lambda *a, **k: str(kmmmu_csv(tmp_path)))
    log = run(kmmmu.kmmmu(), tmp_path, mock("$\\boxed{4}$"), mock("true", "false"))
    m = metrics(log)
    assert m["accuracy"] == pytest.approx(1 / 2)
    assert m["format_failure"] == 0.0 and m["judge_failed"] == 0.0
    scores = [s.scores["answer_judge"] for s in log.samples]
    assert {s.metadata["extracted"] for s in scores} == {"4"} and all(s.metadata["parsed"] for s in scores)


def test_kmmmu_truncated_answer_is_not_judged(tmp_path, monkeypatch):
    pytest.importorskip("math_verify")
    monkeypatch.setattr(kmmmu, "hf_hub_download", lambda *a, **k: str(kmmmu_csv(tmp_path)))
    log = run(kmmmu.kmmmu(), tmp_path, mock("$\\boxed{4}$", stop_reason="max_tokens"), mock("true"))
    assert metrics(log)["accuracy"] == 0.0 and metrics(log)["truncated"] == 1.0
    assert not any(e.event == "model" and e.role == "grader" for s in log.samples for e in s.events)


# --- data (network) -----------------------------------------------------------------------------

@pytest.mark.network
def test_haerae_vision_data():
    try:
        rows = haerae_vision.load_records(haerae_vision.DATASET_REVISION)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dataset not reachable: {exc}")
    samples = [haerae_vision.record_to_sample(r, "original") for r in rows]
    assert len(samples) == 653 and len({s.id for s in samples}) == 653
    assert all(s.metadata["checklist"] for s in samples)
    assert max(len(s.input[1].content) - 1 for s in samples) == 6


@pytest.mark.network
def test_kmmmu_data():
    try:
        samples = kmmmu.load_samples(kmmmu.DATASET_REVISION)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dataset not reachable: {exc}")
    assert len(samples) == 3466
    assert sum(len(s.input[1].content) - 1 for s in samples) == 3707
    assert len({s.metadata["question_type"] for s in samples}) == 5
