"""HLE and IMO-AnswerBench: scoring end to end against ``mockllm``, and the pinned data."""
import hashlib
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput, get_model

from dna_eval.tasks.benchmarks import hle as hle_module
from dna_eval.tasks.benchmarks import imo_answerbench as imo_module
from dna_eval.tasks.benchmarks.hle import SYSTEM_PROMPT, hle
from dna_eval.tasks.benchmarks.imo_answerbench import imo_answerbench


def _judgement(correct, answer="42", confidence=90):
    return ('{"extracted_final_answer": "%s", "reasoning": "because", "correct": "%s", "confidence": %d}'
            % (answer, correct, confidence))


def _mock(replies, seen=None):
    """A ``mockllm`` that answers ``replies`` in order (a ``ModelOutput`` is used as given) and
    records each request's config in ``seen``."""
    it = iter(replies)

    def respond(messages, tools, tool_choice, config):
        if seen is not None:
            seen.append((messages, config))
        r = next(it)
        return r if isinstance(r, ModelOutput) else ModelOutput.from_content("mockllm/model", r)

    return get_model("mockllm/model", custom_outputs=respond)


def _mock_by_prompt(replies, seen=None):
    """A ``mockllm`` whose n-th reply to a prompt containing ``key`` is ``replies[key][n]`` (the last
    one repeats). Samples run concurrently, so replies must not depend on arrival order."""
    count = {k: 0 for k in replies}

    def respond(messages, tools, tool_choice, config):
        text = messages[-1].text
        key = next(k for k in replies if k in text)
        if seen is not None:
            seen.append((key, config, text))
        r = replies[key][min(count[key], len(replies[key]) - 1)]
        count[key] += 1
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, ModelOutput) else ModelOutput.from_content("mockllm/model", r)

    return get_model("mockllm/model", custom_outputs=respond)


def _run(task, subject, grader=None, tmp_path=None):
    roles = {"grader": grader} if grader is not None else None
    return inspect_eval(task, model=subject, model_roles=roles, display="none",
                        log_dir=str(tmp_path / "logs"))[0]


def _metrics(log):
    return {name: m.value for s in log.results.scores for name, m in s.metrics.items()}


# --- HLE ----------------------------------------------------------------------------------------


@pytest.fixture
def tiny_hle(monkeypatch):
    rows = [("q1", "Math", "exactMatch"), ("q2", "Math", "exactMatch"), ("q3", "Physics", "multipleChoice")]
    monkeypatch.setattr(hle_module, "hle_dataset", lambda revision: MemoryDataset([
        Sample(id=i, input=[ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=f"Question {i}")],
               target="42", metadata={"category": c, "answer_type": t}) for i, c, t in rows]))


def test_hle_scores_with_the_grader_and_reports_metrics(tiny_hle, tmp_path):
    seen = []
    subject = _mock(["Explanation: e\nAnswer: 42\nConfidence: 90%"] * 3)
    grader = _mock_by_prompt({"Question q1": [_judgement("yes")], "Question q2": [_judgement("no", "7")],
                              "Question q3": [_judgement("yes")]}, seen)
    log = _run(hle(), subject, grader, tmp_path)
    assert log.status == "success"
    m = _metrics(log)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["Math"] == 0.5 and m["Physics"] == 1.0
    assert m["calibration_error"] != m["calibration_error"]                      # 3 samples: nan (< 100)
    assert m["truncated"] == 0.0 and m["format_failure"] == 0.0
    # the subject saw the official system prompt, then the question
    first = log.samples[0].messages
    assert (first[0].role, first[0].content) == ("system", SYSTEM_PROMPT) and first[1].content == "Question q1"
    # structured output was asked for, and the judge prompt carries question, response and reference
    assert all(c.response_schema is not None for _, c, _t in seen)
    prompt = next(t for k, _, t in seen if k == "Question q1")
    assert "[question]: Question q1" in prompt and "[correct_answer]: 42" in prompt
    assert "[response]: Explanation: e\nAnswer: 42" in prompt


def test_hle_run_settings(tiny_hle):
    task = hle()
    assert task.epochs == 1
    assert task.config.max_tokens is None
    assert task.config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert hle(thinking=False).config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert re.fullmatch(r"[0-9a-f]{40}", hle_module.DATASET_REVISION)


def test_hle_unreadable_judgement_is_retried_then_wrong(tiny_hle, tmp_path):
    seen = []
    grader = _mock_by_prompt({
        "Question q1": ["no idea", _judgement("yes")],                    # second try works
        "Question q2": ["nope", "still nope", "never"],                   # three tries, all unreadable
        "Question q3": [_judgement("no", "None", 100)],
    }, seen)
    log = _run(hle(), _mock(["a"] * 3), grader, tmp_path)
    assert sorted(k for k, *_ in seen) == ["Question q1"] * 2 + ["Question q2"] * 3 + ["Question q3"]
    scores = {s.id: s.scores["hle_judge"] for s in log.samples}
    assert [scores[i].value for i in ("q1", "q2", "q3")] == [1.0, 0.0, 0.0]
    assert "judge_error" in scores["q2"].metadata and scores["q2"].metadata["confidence"] is None
    assert scores["q3"].metadata["extracted"] is None                 # the judge found no answer
    assert _metrics(log)["format_failure"] == pytest.approx(2 / 3)


def test_hle_retry_after_a_failed_call_asks_for_plain_text(tiny_hle, tmp_path):
    seen = []
    grader = _mock_by_prompt({
        "Question q1": [RuntimeError("response_format is not supported"), _judgement("yes")],
        "Question q2": [_judgement("yes")], "Question q3": [_judgement("yes")]}, seen)
    log = _run(hle(), _mock(["a"] * 3), grader, tmp_path)
    assert {s.id: s.scores["hle_judge"].value for s in log.samples}["q1"] == 1.0
    q1 = [c for k, c, _ in seen if k == "Question q1"]
    assert q1[0].response_schema is not None and q1[1].response_schema is None


def test_hle_truncated_answer_scores_zero_without_the_judge(tiny_hle, tmp_path):
    seen = []
    cut = ModelOutput.from_content("mockllm/model", "partial", stop_reason="max_tokens")
    subject = _mock_by_prompt({"Question q1": [cut], "Question q2": ["ok"], "Question q3": ["ok"]})
    log = _run(hle(), subject, _mock([_judgement("yes")] * 2, seen), tmp_path)
    s = {x.id: x.scores["hle_judge"] for x in log.samples}["q1"]
    assert s.value == 0.0 and s.metadata["truncated"] is True
    assert len(seen) == 2
    m = _metrics(log)
    assert m["truncated"] == pytest.approx(1 / 3) and m["accuracy_excl_truncated"] == 1.0


def test_hle_without_a_grader_role_stops(tiny_hle, tmp_path):
    log = _run(hle(), _mock(["a"] * 3), None, tmp_path)
    assert log.status == "error"


# --- IMO-AnswerBench ----------------------------------------------------------------------------

CSV = ('Problem ID,Problem,Short Answer,Category,Subcategory,Source\n'
       'p1,Find $x$ with $2x=6$.,3,Algebra,Equations,Made up\n'
       'p2,Count the ways.,5,Combinatorics,Counting,Made up\n')


@pytest.fixture
def tiny_imo(monkeypatch, tmp_path):
    p = tmp_path / "answerbench.csv"
    p.write_text(CSV, encoding="utf-8")
    monkeypatch.setattr(imo_module, "fetch_csv", lambda commit, sha256: p)


def test_imo_grades_the_final_content_against_the_reference(tiny_imo, tmp_path):
    seen = []
    subject = _mock_by_prompt({"Find $x$": ["so $x=3$: \\boxed{3}"], "Count the": ["I think it is 4: \\boxed{4}"]})
    grader = _mock_by_prompt({"Problem: Find": ["<thinking>...</thinking>\n\\boxed{Correct}"],
                              "Problem: Count": ["\\boxed{Incorrect}"]}, seen)
    log = _run(imo_answerbench(epochs=2), subject, grader, tmp_path)
    assert log.status == "success"
    m = _metrics(log)
    assert m["accuracy"] == 0.5 and m["Algebra"] == 1.0 and m["Combinatorics"] == 0.0
    # subject: instruction then problem; grader: bare problem, the solution and the golden answer
    assert log.samples[0].messages[0].content == "Put your final answer within \\boxed{}.\n\nFind $x$ with $2x=6$."
    prompt = next(t for k, _, t in seen if k == "Problem: Find")
    assert "Problem: Find $x$ with $2x=6$.\n" in prompt and "Model Solution: so $x=3$: \\boxed{3}\n" in prompt
    assert prompt.endswith("Golden Answer: 3")
    assert log.samples[0].scores["imo_answer_grader"].metadata["extracted"] == "3"


def test_imo_run_settings(tiny_imo):
    task = imo_answerbench()
    assert task.epochs == 8 and task.config.max_tokens is None
    assert task.config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert imo_answerbench(epochs=1).epochs == 1
    assert re.fullmatch(r"[0-9a-f]{40}", imo_module.COMMIT) and len(imo_module.CSV_SHA256) == 64


def test_imo_reply_without_a_verdict_is_retried_then_wrong(tiny_imo, tmp_path):
    seen = []
    grader = _mock_by_prompt({
        "Problem: Find": ["The answers look the same.", "\\boxed{Correct}"],   # second try works
        "Problem: Count": ["hmm", "hmm", "hmm"]}, seen)                         # gives up
    log = _run(imo_answerbench(epochs=1), _mock(["\\boxed{3}", "\\boxed{5}"]), grader, tmp_path)
    assert len(seen) == 5
    a, b = (log.samples[0].scores["imo_answer_grader"], log.samples[1].scores["imo_answer_grader"])
    assert log.samples[0].id == "p1"
    assert (a.value, b.value) == (1.0, 0.0) and "judge_error" in b.metadata


def test_imo_no_box_is_a_format_failure_and_truncation_scores_zero(tiny_imo, tmp_path):
    seen = []
    cut = ModelOutput.from_content("mockllm/model", "\\boxed{3", stop_reason="max_tokens")
    log = _run(imo_answerbench(epochs=1), _mock(["it is 3", cut]), _mock(["\\boxed{Correct}"], seen), tmp_path)
    assert len(seen) == 1                                                    # the cut-off answer is not graded
    m = _metrics(log)
    assert m["format_failure"] == 1.0 and m["truncated"] == 0.5 and m["accuracy"] == 0.5
    assert m["accuracy_excl_truncated"] == 1.0


# --- pinned data (network) ----------------------------------------------------------------------


@pytest.mark.network
def test_imo_csv_matches_the_pinned_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("DNA_EVAL_CACHE_DIR", str(tmp_path))
    path = imo_module.fetch_csv()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == imo_module.CSV_SHA256
    samples = imo_module.read_samples(path)
    assert len(samples) == 400 and len({s.id for s in samples}) == 400
    assert all(s.target and s.metadata["problem"] and s.metadata["category"] for s in samples)
    task = imo_answerbench()
    assert len(task.dataset) == 400


@pytest.mark.network
def test_hle_text_only_data_at_the_pinned_revision():
    try:
        task = hle()
    except Exception as e:  # noqa: BLE001  (gated dataset: no token, or the licence not accepted)
        pytest.skip(f"cais/hle is gated and could not be read: {type(e).__name__}: {str(e)[:200]}")
    samples = list(task.dataset)
    assert len(samples) == 2158
    assert all(s.input[0].content == SYSTEM_PROMPT and s.input[1].content and s.target for s in samples)
    assert {s.metadata["answer_type"] for s in samples} == {"exactMatch", "multipleChoice"}
