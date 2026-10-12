"""``letter_choice``, ``shuffled_choices`` and the metrics built for the multiple-choice tasks."""
import asyncio
import math
import re

import pytest
from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, ModelOutput
from inspect_ai.scorer import SampleScore, Score, Target
from inspect_ai.solver import TaskState

from dna_eval.metrics.license_pass import EXAMS, license_pass
from dna_eval.metrics.weighted_accuracy import weighted_accuracy
from dna_eval.scorers.letter_choice import CANDIDATES, EXPECTED, extract_letter, letter_choice
from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED
from dna_eval.solvers.shuffled_choices import permutation
from dna_eval.tasks.benchmarks import gpqa_diamond, kmmlu_pro, kobalt, mmlu_pro

GPQA = [re.compile(gpqa_diamond.ANSWER_PATTERN)]
MMLU = [re.compile(p) for p in mmlu_pro.ANSWER_PATTERNS]
KMMLU = [re.compile(kmmlu_pro.ANSWER_PATTERN)]
KOBALT = [re.compile(kobalt.ANSWER_PATTERN)]


def _letter(text, patterns, letters, **kw):
    return extract_letter(text, patterns, letters, **kw)[0]


def test_last_match_wins_over_first():
    text = "Answer: A\nOn reflection that is wrong.\nAnswer: C"
    assert _letter(text, GPQA, "ABCD") == "C"


def test_gpqa_pattern_forms():
    assert _letter("Answer: $B$", GPQA, "ABCD") == "B"
    assert _letter("answer:\tD", GPQA, "ABCD") == "D"
    assert _letter("Answer : B", GPQA, "ABCD") == "B"
    assert _letter("Answer: E", GPQA, "ABCD") is None       # outside the letter set
    assert _letter("Answer: b", GPQA, "ABCD") is None       # lowercase never equals a letter
    assert _letter("**Answer: B**", GPQA, "ABCD") == "B"
    assert _letter("**Answer:** B", GPQA, "ABCD") is None   # the official pattern does not allow it
    assert _letter("no answer line", GPQA, "ABCD") is None


def test_mmlu_pro_chain_and_bold_stripping_in_scorer():
    chain = lambda t: _letter(t, MMLU, "ABCDEFGHIJ")
    assert chain("so the answer is (J)") == "J"
    assert chain("the answer is B. Wait, the answer is (F)") == "F"          # last match
    assert chain("Answer: G") == "G"                                         # second pattern
    assert chain("Answer: C ... then answer: H") == "H"                      # last, across lines too
    assert chain("answer is (B) but also Answer: D") == "B"                  # first pattern has priority
    assert chain("It is B") is None                                          # no standalone-capital fallback
    assert chain("The answer is K") is None


def test_kmmlu_pro_pattern():
    k = lambda t: _letter(t, KMMLU, "ABCDE")
    assert k("정답: C") == "C"
    assert k("정답 : (B)") == "B"
    assert k("**정답:** E") == "E"                                           # 5-option letter
    assert k("정답: A\n다시 생각해 보니\n정답: D") == "D"
    assert k("최종 정답은 다음과 같습니다.\n정답: B") == "B"
    assert k("정답을 고르겠습니다") is None
    assert k("정답: c") is None                                              # matched, but not a valid letter


def test_kobalt_unique_letter_rule():
    kb = lambda t: extract_letter(t, KOBALT, "ABCDEFGHIJ", unique_letter=True)
    assert kb("풀이 ... 정답은 H입니다.") == ("H", ["H"])
    assert kb("정답은 [H]입니다.") == ("H", ["H"])
    assert kb("정답은 H입니다. 아니, 정답은 H입니다.") == ("H", ["H"])         # same letter repeated is fine
    assert kb("정답은 B입니다. 다시 보면 정답은 C입니다.") == (None, ["B", "C"])   # contradicting statements
    assert kb("정답은 A와 B입니다.") == (None, ["A", "B"])                      # several letters in one statement
    assert kb("정답은 K입니다.") == (None, [])                                 # outside A-J
    assert kb("답은 H") == (None, [])
    assert kb("정답은 입니다") == (None, [])


def test_patterns_need_exactly_one_group():
    with pytest.raises(ValueError):
        letter_choice([r"Answer: [A-D]"])
    with pytest.raises(ValueError):
        letter_choice([r"(Answer): ([A-D])"])


def _score(output: ModelOutput, target="B", metadata=None, **kwargs):
    state = TaskState(model="mockllm/model", sample_id=1, epoch=1, input="q", messages=[], metadata=metadata or {})
    state.output = output
    return asyncio.run(letter_choice(**kwargs)(state, Target(target)))


def _content(text, stop_reason="stop"):
    return ModelOutput.from_content("mockllm/model", text, stop_reason=stop_reason)


def test_scorer_values_and_metadata():
    right = _score(_content("Answer: B"), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert right.value == 1.0 and right.answer == "B"
    assert right.metadata == {TRUNCATED: False, EXTRACTED: "B", EXPECTED: "B"}
    wrong = _score(_content("Answer: C"), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert wrong.value == 0.0 and wrong.metadata[EXTRACTED] == "C" and wrong.metadata[EXPECTED] == "B"
    none = _score(_content("I do not know"), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert none.value == 0.0 and none.metadata[EXTRACTED] is None and none.answer is None


def test_scorer_truncated_scores_zero_but_records_the_answer():
    s = _score(_content("Answer: B", stop_reason="max_tokens"), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert s.value == 0.0 and s.metadata[TRUNCATED] is True and s.metadata[EXTRACTED] == "B"
    s = _score(_content("Answer: B", stop_reason="model_length"), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert s.value == 0.0 and s.metadata[TRUNCATED] is True


def test_scorer_reads_final_content_not_reasoning():
    message = ChatMessageAssistant(content=[ContentReasoning(reasoning="Answer: B"), ContentText(text="Answer: C")])
    s = _score(ModelOutput.from_message(message), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert s.value == 0.0 and s.metadata[EXTRACTED] == "C"
    message = ChatMessageAssistant(content=[ContentReasoning(reasoning="Answer: B"), ContentText(text="")])
    s = _score(ModelOutput.from_message(message), patterns=[gpqa_diamond.ANSWER_PATTERN])
    assert s.metadata[EXTRACTED] is None


def test_scorer_strip_and_target_key_and_candidates():
    s = _score(_content("the answer is **(D)**"), "X", metadata={"right": "D"}, patterns=mmlu_pro.ANSWER_PATTERNS,
               letters="ABCDEFGHIJ", strip=("**",), target_key="right")
    assert s.value == 1.0 and s.metadata[EXPECTED] == "D"
    s = _score(_content("정답은 B입니다. 정답은 C입니다."), "B", patterns=[kobalt.ANSWER_PATTERN],
               letters="ABCDEFGHIJ", unique_letter=True)
    assert s.value == 0.0 and s.metadata[CANDIDATES] == ["B", "C"] and s.metadata[EXTRACTED] is None


def test_permutation_is_deterministic_and_varies_with_epoch_and_sample():
    assert permutation(4, 0, "rec1", 1) == permutation(4, 0, "rec1", 1)
    assert sorted(permutation(4, 0, "rec1", 1)) == [0, 1, 2, 3]
    orders = {tuple(permutation(4, 0, "rec1", e)) for e in range(1, 9)}
    assert len(orders) > 1                                           # not the same order every epoch
    assert len({tuple(permutation(4, seed, "rec1", 1)) for seed in range(12)}) > 1      # the seed matters
    assert len({tuple(permutation(4, 0, f"rec{i}", 1)) for i in range(30)}) > 1        # so does the sample


def _sample_score(value, **metadata):
    return SampleScore(score=Score(value=value), sample_metadata=metadata)


def test_weighted_accuracy():
    scores = [_sample_score(1.0, score=3.0), _sample_score(0.0, score=1.0)]
    assert weighted_accuracy()(scores) == 0.75
    assert weighted_accuracy("w")([_sample_score(1.0, w=1.0), _sample_score(0.0)]) == 0.5       # missing weight = 1
    assert math.isnan(weighted_accuracy()([_sample_score(1.0, score=0.0)]))


def _exam_scores(license_name, subject_accuracy):
    """Four weighted questions per subject, ``subject_accuracy[subject]`` of the points right."""
    out = []
    for subject, acc in subject_accuracy.items():
        out.append(_sample_score(1.0, license_name=license_name, subject=subject, score=acc))
        out.append(_sample_score(0.0, license_name=license_name, subject=subject, score=1.0 - acc))
    return out


def test_license_pass_requires_every_subject_and_the_average():
    subjects = EXAMS["관세사"][0][1]
    good = license_pass()(_exam_scores("관세사", {s: 0.7 for s in subjects}))
    assert good["관세사"] == pytest.approx(0.7) and good["관세사/passed"] == 1.0 and good["licenses_passed"] == 1.0
    low_subject = {s: 0.9 for s in subjects} | {subjects[0]: 0.3}                  # average 0.78, one subject < 40
    assert license_pass()(_exam_scores("관세사", low_subject))["관세사/passed"] == 0.0
    low_average = {s: 0.55 for s in subjects}                                      # all subjects >= 40, average < 60
    assert license_pass()(_exam_scores("관세사", low_average))["관세사/passed"] == 0.0
    assert license_pass()(_exam_scores("관세사", {subjects[0]: 0.9}))["관세사/passed"] == 0.0   # a subject has no questions


def test_license_pass_course_choice_and_special_licenses():
    # 공인노무사: passing either of the two elective courses is enough.
    common = {s: 0.7 for s in ["노동법1", "노동법2", "민법", "사회보험법"]}
    assert license_pass()(_exam_scores("공인노무사", common | {"경영학개론": 0.7}))["공인노무사/passed"] == 1.0
    assert license_pass()(_exam_scores("공인노무사", common | {"경제학원론": 0.7}))["공인노무사/passed"] == 1.0
    assert license_pass()(_exam_scores("공인노무사", common))["공인노무사/passed"] == 0.0
    # 변호사: no per-subject minimum, average line of about 54.2%.
    assert license_pass()(_exam_scores("변호사", {"민사법": 0.6}))["변호사/passed"] == 1.0
    assert license_pass()(_exam_scores("변호사", {"민사법": 0.5}))["변호사/passed"] == 0.0
    # 법무사: subject minimums only, no average line.
    assert license_pass()(_exam_scores("법무사", {s: 0.41 for s in EXAMS["법무사"][0][1]}))["법무사/passed"] == 1.0
    # A licence with no question is not reported.
    assert "의사" not in license_pass()(_exam_scores("관세사", {"회계학": 0.7}))
