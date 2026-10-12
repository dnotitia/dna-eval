"""The judge-graded benchmarks' pieces: prompts, judgement parsing, calibration error, data fetch."""
import hashlib
import urllib.request

import numpy as np
import pytest
from inspect_ai.scorer import Score, SampleScore

from dna_eval.metrics.calibration_error import calibration_error, rms_calibration_error
from dna_eval.prompts import env
from dna_eval.scorers.hle_judge import _extracted, read_judgement
from dna_eval.scorers.imo_answer_grader import last_boxed, read_verdict
from dna_eval.tasks.benchmarks import imo_answerbench as imo

# --- HLE judge prompt -------------------------------------------------------------------------


def test_hle_judge_prompt_is_the_official_text():
    """SHA-256 of the official ``JUDGE_PROMPT.format(...)`` output (typos and all) for this input."""
    out = env.get_template("hle_judge.jinja").render(
        question="What is {x}?\n{{y}}", response="Answer: 4 {z}", correct_answer="4")
    assert hashlib.sha256(out.encode()).hexdigest() == "b0efcf361d572e9e29571ad0962b6e1cf33d76b261423d46003fa3d17fa87b74"
    assert "if there if there is any inconsistency" in out and out.endswith("no confidence score available.")


# --- HLE judgement parsing ----------------------------------------------------------------------

STRUCTURED = ('{"extracted_final_answer": "42", "reasoning": "matches", "correct": "yes", "confidence": 85}')


def test_structured_judgement():
    j = read_judgement(STRUCTURED)
    assert (j.extracted_final_answer, j.correct, j.confidence) == ("42", "yes", 85)


def test_json_inside_prose():
    j = read_judgement("Here is my judgement:\n```json\n" + STRUCTURED.replace("yes", "no") + "\n```\nDone.")
    assert (j.correct, j.confidence) == ("no", 85)


def test_labelled_lines_when_the_server_ignores_the_schema():
    j = read_judgement("extracted_final_answer: x = 3\nreasoning: close but\nnot equal.\n"
                       "**correct**: No\nconfidence: 70%")
    assert (j.extracted_final_answer, j.reasoning, j.correct, j.confidence) == (
        "x = 3", "close but\nnot equal.", "no", 70)


def test_none_answer_is_no_extraction():
    j = read_judgement('{"extracted_final_answer": "None", "reasoning": "r", "correct": "no", "confidence": 100}')
    assert _extracted(j.extracted_final_answer) is None
    assert _extracted(" 42 ") == "42"


@pytest.mark.parametrize("reply", [
    '{"extracted_final_answer": "1", "reasoning": "r", "correct": "yes"}',                  # no confidence
    '{"reasoning": "r", "correct": "yes", "confidence": 5}',                                # no answer
    '{"extracted_final_answer": "1", "reasoning": "r", "correct": "maybe", "confidence": 5}',
    '{"extracted_final_answer": "1", "reasoning": "r", "correct": "yes", "confidence": "high"}',
    "correct: yes",
    "I cannot judge this.",
    "",
])
def test_unreadable_judgements_raise(reply):
    with pytest.raises(ValueError):
        read_judgement(reply)


# --- calibration error --------------------------------------------------------------------------


def _bins(*groups):
    """``(confidence, n, n_correct)`` groups -> arrays, in the order given."""
    conf = np.concatenate([np.full(n, c) for c, n, _ in groups])
    correct = np.concatenate([np.array([True] * k + [False] * (n - k)) for _, n, k in groups])
    return conf, correct


def test_calibration_error_hand_computed_and_last_bin_skipped():
    # Bins of 100: (0.5, 50 right) gap 0; (0.9, 70 right) gap 0.2; last (1.0, 0 right) gap 1.0.
    conf, correct = _bins((0.5, 100, 50), (0.9, 100, 70), (1.0, 100, 0))
    # The official loop skips the last bin: sqrt(100/300 * 0.2**2) = 0.1155
    assert rms_calibration_error(conf, correct) == pytest.approx(np.sqrt(100 / 300 * 0.04))
    # Counting it would give sqrt(100/300 * (0.04 + 1.0)) = 0.5888: the quirk is deliberate.
    assert rms_calibration_error(conf, correct) < 0.2


def test_last_bin_takes_the_remainder_and_is_skipped_with_it():
    # 250 samples: bins [0,100], [100,250]; only the first counts, weighted by 100/250.
    conf, correct = _bins((0.8, 100, 40), (0.9, 150, 0))
    assert rms_calibration_error(conf, correct) == pytest.approx(np.sqrt(100 / 250 * 0.4**2))


def test_calibration_error_small_samples():
    conf, correct = _bins((0.9, 150, 0))
    assert rms_calibration_error(conf, correct) == 0.0           # one bin, skipped
    assert np.isnan(rms_calibration_error(conf[:99], correct[:99]))   # the official script crashes here


def _scored(confidence, ok):
    return SampleScore(score=Score(value=1.0 if ok else 0.0, metadata={"confidence": confidence}))


def test_calibration_metric_rounds_to_two_places_and_skips_unjudged():
    conf, correct = _bins((0.5, 100, 50), (0.9, 100, 70), (1.0, 100, 0))
    scores = [_scored(int(round(c * 100)), ok) for c, ok in zip(conf, correct)]
    scores += [_scored(None, False)] * 7                                   # no judgement: left out
    assert calibration_error()(scores) == 12.0                            # 100 * round(0.1155, 2)


# --- IMO grader ---------------------------------------------------------------------------------


def test_last_verdict_wins():
    assert read_verdict("<thinking>... \\boxed{Incorrect} no wait ...</thinking>\n\\boxed{Correct}") is True
    assert read_verdict("\\boxed{Correct} then \\boxed{Incorrect}") is False
    assert read_verdict("\\boxed{ Correct }") is True


@pytest.mark.parametrize("reply", ["", "Correct", "\\boxed{Maybe}", "\\boxed{correct}", "\\boxed{42}"])
def test_reply_without_a_verdict_raises(reply):
    with pytest.raises(ValueError):
        read_verdict(reply)


def test_last_boxed_balances_braces():
    assert last_boxed("so \\boxed{1} and finally \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert last_boxed("no box") is None
    assert last_boxed("\\boxed{3} then cut off \\boxed{\\frac{1") == "3"


def test_imo_prompt_keeps_the_paper_text_and_adds_section_two():
    out = env.get_template("imo_answer_grader.jinja").render(
        problem="Find $x$.", solution="so \\boxed{3}", golden_answer="3")
    assert out.startswith("# System Role: Deterministic Mathematical Autograder\n")
    assert out.endswith("Problem: Find $x$.\nModel Solution: so \\boxed{3}\nGolden Answer: 3")
    heads = [ln for ln in out.splitlines() if ln.startswith("# ")]
    assert heads == ["# System Role: Deterministic Mathematical Autograder", "# 1. Grading Guidelines (Equivalence Rules)",
                     "# 2. Extraction Protocol", "# 3. Output Protocol (Strict Compliance Required)", "# 4. Input Data"]
    assert "Output exmaple:" in out                      # the paper's own typo
    assert "NOT IN THE PAPER" not in out                 # the note is a template comment


# --- IMO data -----------------------------------------------------------------------------------

CSV = ('Problem ID,Problem,Short Answer,Category,Subcategory,Source\n'
       'imo-bench-algebra-001,"Find all $N$.\nLine two.\n",3,Algebra,Operation,IMO Shortlist 2021\n'
       'imo-bench-combinatorics-002,Count them.,None,Combinatorics,Counting,Made up\n')


def test_read_samples(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(CSV, encoding="utf-8")
    a, b = imo.read_samples(p)
    assert a.id == "imo-bench-algebra-001" and a.target == "3"
    assert a.input == "Put your final answer within \\boxed{}.\n\nFind all $N$.\nLine two."
    assert a.metadata == {"problem": "Find all $N$.\nLine two.", "category": "Algebra",
                          "subcategory": "Operation", "source": "IMO Shortlist 2021"}
    assert b.target == "None"                           # not read as a missing value


def test_fetch_checks_and_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("DNA_EVAL_CACHE_DIR", str(tmp_path))
    calls = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return CSV.encode()

    def urlopen(url, timeout):
        calls.append(url)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    sha = hashlib.sha256(CSV.encode()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        imo.fetch_csv("c0ffee", "0" * 64)
    assert not list(tmp_path.rglob("*.csv"))            # a bad file is not kept
    path = imo.fetch_csv("c0ffee", sha)
    assert imo.fetch_csv("c0ffee", sha) == path
    assert len(calls) == 2 and "/c0ffee/imobench/answerbench_v2.csv" in calls[-1]
    with pytest.raises(ValueError, match="SHA-256"):    # the cache is checked too
        imo.fetch_csv("c0ffee", "0" * 64)
