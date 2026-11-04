"""The multiple-choice benchmark tasks: prompts, and a full run of each on a mock model with a tiny dataset.

The network tests load the real datasets at their pinned revisions; they skip when the
Hub is unreachable or the dataset is gated and no access is granted.
"""
import itertools
import json

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, get_model

from dna_eval.tasks.benchmarks import click, haerae_bench, kmmlu_pro, kmmlu_redux, kobalt


def mock(*replies):
    outputs = [r if isinstance(r, ModelOutput) else ModelOutput.from_content("mockllm/model", r) for r in replies]
    return get_model("mockllm/model", custom_outputs=itertools.cycle(outputs))


def run(task, tmp_path, *replies):
    log = inspect_eval(task, model=mock(*replies), display="none", log_dir=str(tmp_path / "logs"))[0]
    assert log.status == "success", log.error
    return log


def metrics(log):
    """Every metric value of the run by name (a metric that returns several values has its own entry)."""
    return {name: m.value for score in log.results.scores for name, m in score.metrics.items()}


def prompt_of(sample):
    return next(m.text for m in sample.messages if m.role == "user")


# --- KMMLU-Pro ----------------------------------------------------------------------------------

KMMLU_ROWS = [
    {"question": "1+1은?", "options": ["1", "2", "3", "4"], "solution": 2, "score": 3.0,
     "license_name": "관세사", "subject": "회계학"},
    {"question": "2+2는?", "options": ["1", "2", "3", "4", "5"], "solution": "5", "score": 1.0,
     "license_name": "관세사", "subject": "무역영어"},
    {"question": "3+3은?", "options": ["6", "7", "8", "9"], "solution": 1, "score": 2.0,
     "license_name": "약사", "subject": "생명약학"},
]


def test_kmmlu_pro_prompts_are_verbatim():
    four = kmmlu_pro.build_prompt("Q?", ["a", "b", "c", "d"])
    assert four == ("다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 ABCD 중 하나이고, \"정답:\" 뒤에 와야 합니다. "
                    "정답을 고르기 전에 차근차근 생각하고 추론하세요.\n \nQ?\n\nA) a\nB) b\nC) c\nD) d")
    five = kmmlu_pro.build_prompt("Q?", ["a", "b", "c", "d", "e"])
    assert "ABCDE 중 하나" in five and five.endswith("D) d\nE) e")
    with pytest.raises(ValueError):
        kmmlu_pro.build_prompt("Q?", ["a", "b"])
    s = kmmlu_pro.record_to_sample(KMMLU_ROWS[1], 1)
    assert s.target == "E" and s.metadata == {"license_name": "관세사", "subject": "무역영어", "score": 1.0}


def test_kmmlu_pro_end_to_end_weighted(tmp_path, monkeypatch):
    path = tmp_path / "kmmlu_pro.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in KMMLU_ROWS) + "\n", encoding="utf-8")
    monkeypatch.setattr(kmmlu_pro, "hf_hub_download", lambda *a, **k: str(path))
    # Always answers B: right for the first question (weight 3) only.
    log = run(kmmlu_pro.kmmlu_pro(), tmp_path, "차근차근 풀면 이렇습니다.\n정답: B")
    m = metrics(log)
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["weighted_accuracy"] == pytest.approx(3 / 6)
    assert m["license_name/관세사"] == pytest.approx(3 / 4) and m["license_name/약사"] == 0.0
    assert m["license_passed/관세사"] == 0.0 and m["licenses_passed"] == 0.0           # a licence needs every subject
    assert log.results.scores[0].metrics["weighted_accuracy"].value == m["weighted_accuracy"]
    assert log.eval.task_args["revision"] == kmmlu_pro.DATASET_REVISION
    assert log.results.headline.metric == "weighted_accuracy"


# --- KMMLU-Redux --------------------------------------------------------------------------------

REDUX_ROWS = [
    {"question": "1+1은?", "options": ["1", "2", "3", "4"], "solution": "2", "category": "건설", "license_name": "조경기사"},
    {"question": "2+2는?", "options": ["1", "2", "3", "4"], "solution": "4", "category": "기계", "license_name": "기계기사"},
]


def test_kmmlu_redux_uses_the_kmmlu_pro_prompt():
    s = kmmlu_redux.record_to_sample(REDUX_ROWS[1], 1)
    assert s.input == kmmlu_pro.build_prompt("2+2는?", ["1", "2", "3", "4"])
    assert s.target == "D" and s.metadata == {"category": "기계", "license_name": "기계기사"}


def test_kmmlu_redux_end_to_end(tmp_path, monkeypatch):
    path = tmp_path / "kmmlu_redux.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in REDUX_ROWS) + "\n", encoding="utf-8")
    monkeypatch.setattr(kmmlu_redux, "hf_hub_download", lambda *a, **k: str(path))
    log = run(kmmlu_redux.kmmlu_redux(), tmp_path, "풀이 ...\n정답: B")      # right for the first question only
    m = metrics(log)
    assert m["accuracy"] == pytest.approx(1 / 2)
    assert m["category/건설"] == 1.0 and m["category/기계"] == 0.0
    assert log.results.headline.metric == "accuracy"
    assert log.eval.task_args["revision"] == kmmlu_redux.DATASET_REVISION


# --- HAE-RAE Bench ------------------------------------------------------------------------------

HAERAE_ROWS = {
    "history": [{"query": "다음 질문을 읽고 정답으로 가장 알맞은 것을 고르시요.\n### 질문:\nQ\n\n### 선택지:\n"
                          "(A) a\n(B) b\n(C) c\n(D) d\n(E) e\n\n### 정답:", "options": "['a','b','c','d','e']",
                 "answer": "(B)"}],
    "csat_law": [{"query": "다음 사례에 대한 옳은 판단은?\n### 선택지: (A) ㄱ (B) ㄴ (C) ㄷ (D) ㄹ (E) ㅁ",
                  "options": "ㄱ | ㄴ | ㄷ | ㄹ | ㅁ", "answer": "(E)"}],
}


def test_haerae_prompt_keeps_the_query_and_reads_the_letter():
    s = haerae_bench.record_to_sample(HAERAE_ROWS["history"][0], "history", 0)
    assert s.input == haerae_bench.INSTRUCTION + "\n\n" + HAERAE_ROWS["history"][0]["query"]
    assert s.target == "B" and s.id == "history-0" and s.metadata == {"subset": "history"}
    with pytest.raises(ValueError):
        haerae_bench.record_to_sample({"query": "q", "answer": "It's without"}, "lyrics_denoising", 0)
    with pytest.raises(ValueError):
        haerae_bench.load_samples(["lyrics_denoising"], haerae_bench.DATASET_REVISION)


def test_haerae_end_to_end(tmp_path, monkeypatch):
    calls = []

    def fake_hf_dataset(repo, name, split, revision, sample_fields):
        calls.append((repo, name, split, revision))
        return [sample_fields(r) for r in HAERAE_ROWS[name]]

    monkeypatch.setattr(haerae_bench, "hf_dataset", fake_hf_dataset)
    log = run(haerae_bench.haerae_bench(subsets=["history", "csat_law"]), tmp_path, "생각해 보면...\n\n정답: (B)")
    m = metrics(log)
    assert m["accuracy"] == pytest.approx(1 / 2)
    assert m["subset/history"] == 1.0 and m["subset/csat_law"] == 0.0
    assert calls == [(haerae_bench.REPO, n, "test", haerae_bench.DATASET_REVISION) for n in ("history", "csat_law")]


# --- CLIcK --------------------------------------------------------------------------------------

CLICK_RECORDS = [
    {"id": "KIIP_economy_1", "paragraph": "", "question": "경제 질문", "choices": ["가", "나", "다", "라"], "answer": "나"},
    {"id": "CSAT_korean_22_5", "paragraph": "지문", "question": "문법 질문",
     "choices": ["가", "나", "다", "라", "마"], "answer": "마"},
    {"id": "TK_2016_2", "paragraph": "", "question": "기능 질문", "choices": ["가", "나", "다", "라"], "answer": "가"},
]


def test_click_prompts_and_groups():
    plain = click.record_to_sample(CLICK_RECORDS[0])
    assert plain.input == ("주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오. "
                           "최종 정답은 \"정답:\" 뒤에 알파벳 하나로 쓰시오.\n\n질문: 경제 질문\n보기:\n"
                           "A:가, B: 나, C: 다, D: 라")
    assert plain.target == "B"
    assert plain.metadata == {"category": "Culture", "subcategory": "economy", "source_id": "KIIP_economy_1"}
    five = click.record_to_sample(CLICK_RECORDS[1])
    assert five.input.startswith("주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D, E 중에 골라")
    assert "맥락: 지문\n질문: 문법 질문\n보기:\nA:가, B: 나, C: 다, D: 라, E: 마" in five.input
    assert five.target == "E" and five.metadata["category"] == "Language" and five.metadata["subcategory"] == "textual"
    assert click.record_to_sample(CLICK_RECORDS[2]).metadata["subcategory"] == "grammar"


@pytest.mark.parametrize("record_id, question, expected", [
    ("CSAT_korean_22_30", "q", "functional"),       # question number > 34 only for the 23 exam; see below
    ("CSAT_korean_23_40", "q", "functional"),
    ("CSAT_korean_23_20", "q", "textual"),
    ("CSAT_korean_15_12", "q", "grammar"),
    ("CSAT_korean_15_3", "q", "functional"),
    ("Kedu_16_4", "대화 속 발화", "functional"),
    ("Kedu_16_5", "문법", "grammar"),
    ("PSE_korean_1", "q", "functional"),
    ("TK_2019_30", "q", "textual"),
    ("KHB_66_1", "q", "history"),
    ("PSAT_17_2", "q", "law"),
    ("Kedu_Society_3", "q", "society"),
    ("KIIP_popular_3", "q", "popular"),
])
def test_click_subcategory_rules(record_id, question, expected):
    if record_id == "CSAT_korean_22_30":
        expected = "textual"
    assert click.subcategory({"id": record_id, "question": question}) == expected


def test_click_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(click, "hf_dataset", lambda *a, sample_fields, **k:
                        MemoryDataset([sample_fields(r) for r in CLICK_RECORDS]))
    log = run(click.click(), tmp_path, "생각해 보면...\n정답: B")                       # right for the first only
    m = metrics(log)
    assert m["accuracy"] == pytest.approx(1 / 3)
    assert m["category/Culture"] == 1.0 and m["category/Language"] == 0.0
    assert m["subcategory/economy"] == 1.0 and m["subcategory/textual"] == 0.0
    log = run(click.click(), tmp_path / "e", "정답: E")                                 # the 5th option is a valid answer
    assert metrics(log)["subcategory/textual"] == 1.0 and metrics(log)["format_failure"] == 0.0


# --- KoBALT -------------------------------------------------------------------------------------

KOBALT_RECORDS = [
    {"ID": "k1", "Class": "Syntax", "Subclass": "Agreement", "Level": 1, "Sampling_YN": 0,
     "Question": "지문 하나\n\nA: 가\nB: 나\nC: 다\n", "Answer": "B"},
    {"ID": "k2", "Class": "Semantics", "Subclass": "Idioms", "Level": 3, "Sampling_YN": 0,
     "Question": "지문 둘\n\nA: 가\nB: 나\nC: 다\n", "Answer": "C "},
]


def test_kobalt_prompt_follows_the_protocol_json():
    s = kobalt.record_to_sample(KOBALT_RECORDS[0])
    assert s.input == ("다음 문제에 대해서 충분히 생각하고 추론하여, 10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.\n\n"
                       "지문 하나\n\nA: 가\nB: 나\nC: 다\n\n\n"
                       "답변은 반드시 다음 형식을 엄격히 지켜야 합니다: \"정답은 [정답 보기]입니다.\"로 끝나야 하고, "
                       "[정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다.\n"
                       "정답: 문제를 풀기 위해, 한 번 천천히 생각해봅시다.")
    assert kobalt.record_to_sample(KOBALT_RECORDS[1]).target == "C"            # the answer is stripped


def test_kobalt_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(kobalt, "hf_dataset", lambda *a, sample_fields, **k:
                        MemoryDataset([sample_fields(r) for r in KOBALT_RECORDS]))
    log = run(kobalt.kobalt(), tmp_path, "풀이.\n정답은 B입니다.")
    assert log.samples[0].messages[0].role == "system" and log.samples[0].messages[0].text == kobalt.SYSTEM
    m = metrics(log)
    assert m["accuracy"] == 0.5 and m["class/Syntax"] == 1.0 and m["class/Semantics"] == 0.0
    assert m["level/1"] == 1.0 and m["level/3"] == 0.0
    # Two different letters in two statements: wrong, and recorded as a format failure.
    log = run(kobalt.kobalt(), tmp_path / "two", "정답은 B입니다. 다시 보니 정답은 C입니다.")
    m = metrics(log)
    assert m["accuracy"] == 0.0 and m["format_failure"] == 1.0
    assert all(s.scores["letter_choice"].metadata["candidates"] == ["B", "C"] for s in log.samples)
    log = run(kobalt.kobalt(), tmp_path / "none", "잘 모르겠습니다.")
    assert metrics(log)["accuracy"] == 0.0 and metrics(log)["format_failure"] == 1.0


# --- the real datasets, at the pinned revisions ---------------------------------------------------

def _skip_if_unreachable(exc: Exception):
    text = f"{type(exc).__name__}: {exc}"
    if any(k in text for k in ("GatedRepo", "403", "401", "Gated", "Connection", "offline", "Max retries")):
        pytest.skip(f"dataset not reachable (gated or offline): {text[:160]}")
    raise exc


@pytest.mark.network
def test_kmmlu_pro_data():
    try:
        samples = kmmlu_pro.load_samples(kmmlu_pro.DATASET_REVISION)
    except Exception as exc:
        _skip_if_unreachable(exc)
    assert len(samples) == 2822
    assert {s.target for s in samples} <= set("ABCDE")
    assert {s.metadata["license_name"] for s in samples} <= set(kmmlu_pro_licenses())
    assert all(s.metadata["score"] > 0 for s in samples)


def kmmlu_pro_licenses():
    from dna_eval.metrics.license_pass import EXAMS
    return EXAMS


@pytest.mark.network
def test_click_data():
    from inspect_ai.dataset import hf_dataset
    try:
        samples = list(hf_dataset("EunsuKim/CLIcK", split="train", revision=click.DATASET_REVISION,
                                  sample_fields=click.record_to_sample, auto_id=True))
    except Exception as exc:
        _skip_if_unreachable(exc)
    assert len(samples) == 1995 and len({s.id for s in samples}) == 1995       # numbered by the loader
    assert len({s.metadata["source_id"] for s in samples}) == 1987              # the data's own ids repeat
    assert sum(1 for s in samples if s.target == "E") > 0
    assert len({s.metadata["subcategory"] for s in samples}) == 11
    assert sum(s.metadata["category"] == "Language" for s in samples) == 285 + 233 + 132


@pytest.mark.network
def test_kobalt_data():
    from inspect_ai.dataset import hf_dataset
    try:
        samples = list(hf_dataset("snunlp/KoBALT-700", name="kobalt_v1", split="raw", revision=kobalt.DATASET_REVISION,
                                  sample_fields=kobalt.record_to_sample))
    except Exception as exc:
        _skip_if_unreachable(exc)
    assert len(samples) == 700 and len({s.id for s in samples}) == 700
    assert {s.target for s in samples} <= set("ABCDEFGHIJ")
    assert {s.metadata["level"] for s in samples} == {1, 2, 3}
    assert len({s.metadata["class"] for s in samples}) == 5


@pytest.mark.network
def test_kmmlu_redux_data():
    try:
        samples = kmmlu_redux.load_samples(kmmlu_redux.DATASET_REVISION)
    except Exception as exc:
        _skip_if_unreachable(exc)
    assert len(samples) == 2587
    assert {s.target for s in samples} == set("ABCD")
    assert len({s.metadata["category"] for s in samples}) == 14


@pytest.mark.network
def test_haerae_bench_data():
    try:
        samples = haerae_bench.load_samples(list(haerae_bench.SUBSETS), haerae_bench.DATASET_REVISION)
    except Exception as exc:
        _skip_if_unreachable(exc)
    assert len(samples) == 3606 and len({s.id for s in samples}) == 3606
    assert {s.target for s in samples} == set("ABCDE")
    assert {s.metadata["subset"] for s in samples} == set(haerae_bench.SUBSETS)
