"""KMMLU-HARD, KorMedMCQA and KBL: prompts, answer reading, metrics, and full runs on a mock model."""
import itertools
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Score

from dna_eval.metrics.macro_accuracy import macro_accuracy
from dna_eval.scorers.letter_choice import extract_letter
from dna_eval.tasks.benchmarks import kbl, kmmlu_hard, kormedmcqa


def run(task, tmp_path, *replies):
    outs = [ModelOutput.from_content("mockllm/model", r) for r in replies]
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=itertools.cycle(outs)),
                       display="none", log_dir=str(tmp_path / "logs"))[0]
    assert log.status == "success", log.error
    return log


def metrics(log):
    return {name: m.value for s in log.results.scores for name, m in s.metrics.items()}


def fake_hf(rows):
    """An ``hf_dataset`` stand-in serving ``rows[(config, split)]``."""
    def hf_dataset(repo, name, split, revision, sample_fields):
        return [sample_fields(r) for r in rows[(name, split)]]
    return hf_dataset


def _score(value, **md):
    return SampleScore(score=Score(value=value), sample_metadata=md)


def test_macro_accuracy_weights_groups_equally():
    scores = [_score(1.0, g="a"), _score(1.0, g="a"), _score(1.0, g="a"), _score(0.0, g="b")]
    assert macro_accuracy("g")(scores) == pytest.approx(0.5)          # micro would be 0.75


# --- KorMedMCQA / KMMLU-HARD ------------------------------------------------------------------

def test_kormedmcqa_cascade():
    pats = [re.compile(p) for p in kormedmcqa.choice_patterns("ABCDE")]
    read = lambda t: extract_letter(t.replace("**", ""), pats, "ABCDE", uppercase=True)[0]  # noqa: E731
    assert read("정답: B") == "B"
    assert read("풀이 ... 정답은 C입니다") == "C"
    assert read("정답 d") == "D"                               # case-insensitive, read upper-cased
    assert read("A. 틀림, 그래서 E. 입니다") == "E"            # the third pattern, last match
    assert read("답 없음") is None
    # The prompt's full-width colon, in bold, then an explanation naming other letters.
    assert read("**정답： A**\n\n*   **B (틀림):** ... D와 E도 아님") == "A"
    assert read("정답： **C. 설명**\n\n해설: E는 아님") == "C"


MED = {"question": " 질문 ", "A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}


def test_kormedmcqa_prompt_is_five_shot():
    shots = [dict(MED, question=f"예제{i}", answer=i) for i in (1, 2)]
    p = kormedmcqa.build_prompt(dict(MED, answer=3), shots, "ABCDE")
    assert p == ("예제1\nA. a\nB. b\nC. c\nD. d\nE. e\n정답： A\n\n"
                 "예제2\nA. a\nB. b\nC. c\nD. d\nE. e\n정답： B\n\n"
                 "질문\nA. a\nB. b\nC. c\nD. d\nE. e\n정답：")


def test_kormedmcqa_end_to_end(tmp_path, monkeypatch):
    rows = {(exam, "fewshot"): [dict(MED, answer=1)] * 5 for exam in kormedmcqa.EXAMS}
    rows[("doctor", "test")] = [dict(MED, answer=2, year=2020)]
    rows[("nurse", "test")] = [dict(MED, answer=4, year=2021)]
    monkeypatch.setattr(kormedmcqa, "hf_dataset", fake_hf(rows))
    log = run(kormedmcqa.kormedmcqa(exams=["doctor", "nurse"]), tmp_path, "정답: B")
    m = metrics(log)
    assert m["accuracy"] == 0.5 and m["exam/doctor"] == 1.0 and m["exam/nurse"] == 0.0
    assert log.samples[0].input.count("정답： A") == 5


def test_kmmlu_hard_end_to_end_macro(tmp_path, monkeypatch):
    q = {"question": "문제", "A": "1", "B": "2", "C": "3", "D": "4", "category": "x", "cot": ""}
    rows = {("accounting", "dev"): [dict(q, answer=1)] * 5, ("math", "dev"): [dict(q, answer=1)] * 5,
            ("accounting", "test"): [dict(q, answer=2)] * 3, ("math", "test"): [dict(q, answer=3)]}
    monkeypatch.setattr(kmmlu_hard, "hf_dataset", fake_hf(rows))
    log = run(kmmlu_hard.kmmlu_hard(subjects=["accounting", "math"]), tmp_path, "생각해 보면 정답은 B입니다")
    m = metrics(log)
    assert m["accuracy"] == pytest.approx(3 / 4) and m["macro_accuracy"] == pytest.approx(0.5)
    assert log.results.headline.metric == "macro_accuracy"
    assert "E." not in log.samples[0].input                    # four options


# --- KBL --------------------------------------------------------------------------------------

def test_kbl_task_set_and_prompts():
    assert len(kbl.TASKS) == 65 and len({t for t, *_ in kbl.TASKS}) == 65
    assert kbl.DESCRIPTION.endswith("입니다.\\n")                  # a backslash and an "n", as the YAML parses
    s = kbl.record_to_sample({"question": "Q", "A": "a", "B": "b", "C": "c", "D": "d", "E": "e", "label": "B"},
                             "kbl_knowledge_legal_concept_qa", "legal_concept_qa", "label", "ABCDE", 0)
    assert s.input == (kbl.DESCRIPTION + "### 질문: Q\nA. a\nB. b\nC. c\nD. d\nE. e\n"
                       "'A', 'B', 'C', 'D', 'E' 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.")
    assert s.target == "B" and s.metadata == {"task": "kbl_knowledge_legal_concept_qa", "family": "knowledge",
                                              "letters": "ABCDE"}
    assert [t for t in kbl.TASKS if t[0] == "bar_exam_responsibility_2015"][0][3] == "ABCD"


def test_kbl_reads_the_first_accepted_capital():
    assert kbl.first_letter("답변: B", "ABCDE") == "B"
    assert kbl.first_letter("정답은 C이고 D는 아닙니다", "ABCDE") == "C"
    assert kbl.first_letter("E는 틀렸고 답변: A", "ABCD") == "A"            # E is not an option there
    assert kbl.first_letter("모르겠습니다", "ABCDE") is None


def test_kbl_end_to_end_with_paper_averages(tmp_path, monkeypatch):
    q = {"question": "Q", "A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}
    knowledge = [t for t, *_ in kbl.TASKS if t.startswith("kbl_knowledge_")]
    rows = {(t, "test"): [dict(q, label="A")] for t in knowledge}
    rows[("kbl_knowledge_legal_concept_qa", "test")] = [dict(q, label="B")]
    monkeypatch.setattr(kbl, "hf_dataset", fake_hf(rows))
    log = run(kbl.kbl(families=["knowledge"]), tmp_path, "답변: A")
    m = metrics(log)
    assert m["knowledge_avg"] == pytest.approx(6 / 7) and "bar_exam_2024_avg" not in m
    assert m["task/kbl_knowledge_legal_concept_qa"] == 0.0 and m["family/knowledge"] == pytest.approx(6 / 7)


# --- data (network) ---------------------------------------------------------------------------

def _skip(exc):
    pytest.skip(f"dataset not reachable: {str(exc)[:160]}")


@pytest.mark.network
def test_kmmlu_hard_data():
    try:
        samples = kmmlu_hard.load_samples(list(kmmlu_hard.SUBJECTS), 5, kmmlu_hard.DATASET_REVISION)
    except Exception as exc:  # noqa: BLE001
        _skip(exc)
    assert len(samples) == 4104 and {s.target for s in samples} == set("ABCD")
    assert len({s.metadata["subject"] for s in samples}) == 45


@pytest.mark.network
def test_kormedmcqa_data():
    try:
        samples = kormedmcqa.load_samples(list(kormedmcqa.EXAMS), 5, kormedmcqa.DATASET_REVISION)
    except Exception as exc:  # noqa: BLE001
        _skip(exc)
    counts = {e: sum(s.metadata["exam"] == e for s in samples) for e in kormedmcqa.EXAMS}
    assert counts == {"doctor": 435, "nurse": 878, "pharm": 885, "dentist": 811}


@pytest.mark.network
def test_kbl_data():
    try:
        samples = kbl.load_samples(list(kbl.FAMILIES), kbl.DATASET_REVISION)
    except Exception as exc:  # noqa: BLE001
        _skip(exc)
    counts = {f: sum(s.metadata["family"] == f for s in samples) for f in kbl.FAMILIES}
    assert counts == {"knowledge": 510, "reasoning": 288, "bar_exam": 2578}
    assert {s.target for s in samples} == set("ABCDE") | {"정답없음"}
