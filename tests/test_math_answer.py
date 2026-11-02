"""Math answer reading and comparison: the helpers in ``utils/math_answer.py`` and the
``math_answer`` scorer, run end to end against ``mockllm`` on a small in-memory dataset."""
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.solver import generate

from dna_eval.scorers.math_answer import math_answer
from dna_eval.utils.math_answer import (extract_boxed_answer, is_equiv_math_verify, is_equiv_string,
                                        last_boxed, last_integer, parse_integer)


# --- boxed extraction ---------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    (r"so \boxed{42}.", "42"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),                       # nested braces
    (r"\boxed{\text{a}^{2}} and then", r"\text{a}^{2}"),
    (r"first \boxed{1}, then \boxed{2}", "2"),                      # several boxes: the last
    (r"\fbox{7}", "7"),
    (r"\boxed{3} later \fbox{9}", "9"),                             # last of either kind
    (r"\boxed{\boxed{5}}", "5"),                                    # box inside a box
    (r"\boxed {8}", "8"),
    (r"\boxed{1} and a cut-off \boxed{2", "1"),                     # unclosed box ignored
    ("no box here, 17 apples", None),
    ("", None),
])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


def test_last_integer():
    assert last_integer("we get 12 and then 345 apples") == 345
    assert last_integer("x = 3.5") == 5                                    # a decimal is two numbers
    assert last_integer("no digits") is None
    assert last_integer("abc123") is None                                  # needs word boundaries


@pytest.mark.parametrize("s, expected", [
    ("42", 42), ("042", 42), ("42.0", 42), (" 42 ", 42), ("$42$", 42), (r"\text{42}", 42),
    ("1,000", 1000), ("-7", -7), ("42.5", None), (r"\frac{1}{2}", None), ("", None), ("abc", None),
])
def test_parse_integer(s, expected):
    assert parse_integer(s) == expected


# --- the HRM8K answer string ----------------------------------------------------------------

@pytest.mark.parametrize("response, expected", [
    (r"... $\boxed{12}$", "12"),
    (r"\boxed{x = 5}", "5"),                                        # text after the last =
    (r"\boxed{a=1, b=2}", "2"),
    (r"first \boxed{1} then \boxed{\frac{1}{6}}", r"\frac{1}{6}"),
    (r"\boxed{3", None),                                            # box never closed: no fallback
    ("the answer is $x + 1 = 7$ done", "7"),                         # no box: last $...$ span
    ("so the total is 7 plus 8 equals 15.", "15"),                   # no box, no $: after the last =
    ("it costs 12 dollars, or 15", "15"),                            # no box, no =: last number
    ("nothing numeric", None),
    ("", None),
])
def test_extract_boxed_answer(response, expected):
    assert extract_boxed_answer(response) == expected


# --- equivalence --------------------------------------------------------------------------

@pytest.mark.parametrize("answer, gold, string_eq", [
    ("42", "42", True), ("42.0", "42", True),
    (r"\dfrac{1}{6}", r"\frac{1}{6}", True),                        # normalised by the Hendrycks rules
    ("1/6", r"\frac{1}{6}", True),
    ("0.5", r"\frac12", True),
    (r"\frac{1}{6}", r"\frac{1}{6}", True),
    (r"8\pi", r"8\pi", True),                                       # a non-numeric reference stays intact
    ("8", r"8\pi", False),
    ("43", "42", False),
    (None, "42", False),
])
def test_is_equiv_string(answer, gold, string_eq):
    assert is_equiv_string(answer, gold) is string_eq


@pytest.mark.parametrize("answer, gold, equal", [
    ("1/6", r"\frac{1}{6}", True),
    (r"8\pi", r"8\pi", True),
    (r"8 \pi", r"8\pi", True),
    ("0.5", r"\frac12", True),
    (r"\frac{\sqrt{2}}{2}", r"\frac{1}{\sqrt{2}}", True),
    (r"\{1, 2\}", r"\{2, 1\}", True),                               # sets
    ("(1, 2]", "(1, 2]", True),                                     # intervals
    ("(1, 2]", "(1, 2)", False),
    (r"\frac{1}{6}", r"\frac{1}{5}", False),
    ("7", "8", False),
    ("", "8", False),
    (None, "8", False),
])
def test_is_equiv_math_verify(answer, gold, equal):
    assert is_equiv_math_verify(answer, gold) is equal


# --- the scorer, end to end ------------------------------------------------------------------

def _run(mode, target, reply, tmp_path, stop_reason="stop", epochs=1):
    out = ModelOutput.from_content("mockllm/model", reply)
    out.choices[0].stop_reason = stop_reason
    model = get_model("mockllm/model", custom_outputs=[out] * epochs)
    task = Task(dataset=MemoryDataset([Sample(input="q", target=target, metadata={"category": "c"})]),
                solver=generate(), scorer=math_answer(mode), epochs=epochs)
    log = inspect_eval(task, model=model, display="none", log_dir=str(tmp_path))[0]
    assert log.status == "success", log.error
    return log, log.samples[0].scores["math_answer"]


@pytest.mark.parametrize("reply, target, value, extracted", [
    (r"The answer is \boxed{042}.", "42", 1.0, "042"),
    (r"\boxed{42.0}", "42", 1.0, "42.0"),
    (r"first \boxed{1} finally \boxed{204}", "204", 1.0, "204"),
    ("I get 12 and so 204", "204", 1.0, "204"),                 # no box: last integer
    (r"\boxed{205}", "204", 0.0, "205"),
    (r"\boxed{\frac{1}{2}}", "204", 0.0, r"\frac{1}{2}"),
    ("no number at all", "204", 0.0, None),
])
def test_integer_mode(reply, target, value, extracted, tmp_path):
    _, score = _run("integer", target, reply, tmp_path)
    assert score.value == value
    assert score.metadata == {"truncated": False, "extracted": extracted, "boxed": r"\boxed" in reply}


@pytest.mark.parametrize("reply, target, value", [
    (r"$\boxed{\frac{1}{6}}$", r"\frac{1}{6}", 1.0),
    (r"$\boxed{1/6}$", r"\frac{1}{6}", 1.0),
    (r"$\boxed{0.5}$", r"\frac{1}{2}", 1.0),
    (r"$\boxed{8 \pi}$", r"8\pi", 1.0),                          # equal for math-verify only
    (r"$\boxed{\{2, 1\}}$", r"\{1, 2\}", 1.0),
    (r"$\boxed{x = 15}$", "15", 1.0),
    (r"$\boxed{16}$", "15", 0.0),
    (r"$\boxed{8}$", r"8\pi", 0.0),
])
def test_expression_mode(reply, target, value, tmp_path):
    _, score = _run("expression", target, reply, tmp_path)
    assert score.value == value


def test_expression_mode_without_an_answer(tmp_path):
    _, score = _run("expression", "15", "I could not work it out", tmp_path)
    assert score.value == 0.0 and score.metadata["extracted"] is None and score.metadata["boxed"] is False


def test_truncated_response_scores_zero_and_is_flagged(tmp_path):
    log, score = _run("integer", "204", r"so \boxed{204}", tmp_path, stop_reason="max_tokens")
    assert score.value == 0.0
    assert score.metadata == {"truncated": True, "extracted": "204", "boxed": True}
    diag = {k: m.value for sc in log.results.scores for k, m in sc.metrics.items()}
    assert diag["truncated"] == 1.0 and diag["format_failure"] == 0.0


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        math_answer("fraction")


def test_reads_final_content_not_reasoning(tmp_path):
    message = ChatMessageAssistant(
        content=[ContentReasoning(reasoning=r"maybe \boxed{3}"), ContentText(text="The answer is 12")])
    out = ModelOutput(model="mockllm/model", choices=[ChatCompletionChoice(message=message, stop_reason="stop")])
    model = get_model("mockllm/model", custom_outputs=[out])
    task = Task(dataset=MemoryDataset([Sample(input="q", target="3")]), solver=generate(), scorer=math_answer("integer"))
    log = inspect_eval(task, model=model, display="none", log_dir=str(tmp_path))[0]
    score = log.samples[0].scores["math_answer"]
    assert score.value == 0.0 and score.metadata["extracted"] == "12"


def test_diagnostics_cover_every_run(tmp_path):
    first = ModelOutput.from_content("mockllm/model", r"\boxed{1}")
    second = ModelOutput.from_content("mockllm/model", r"\boxed{1}")
    second.choices[0].stop_reason = "max_tokens"
    task = Task(dataset=MemoryDataset([Sample(input="q", target="1")]), solver=generate(),
                scorer=math_answer("integer"), epochs=2)
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=[first, second]),
                       display="none", log_dir=str(tmp_path))[0]
    metrics = {k: m.value for s in log.results.scores for k, m in s.metrics.items()}
    assert metrics["accuracy"] == 0.5 and metrics["truncated"] == 0.5 and metrics["accuracy_excl_truncated"] == 1.0


def test_expression_mode_names_the_math_extra_when_math_verify_is_missing(monkeypatch):
    import sys

    import pytest

    from dna_eval.scorers.math_answer import math_answer

    monkeypatch.setitem(sys.modules, "math_verify", None)
    with pytest.raises(RuntimeError, match=r'pip install "dna-eval\[math\]"'):
        math_answer("expression")
    math_answer("integer")          # the integer mode needs no math-verify
