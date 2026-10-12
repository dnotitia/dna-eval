"""Instruction following: strict and loose checks, the pooled metric, and the scorer end to end."""
import itertools

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.solver import generate

from dna_eval.metrics.instruction_following import instruction_accuracy
from dna_eval.scorers import instruction_following as scorer_module
from dna_eval.scorers.instruction_following import check_instruction, instruction_following, loose_variants

LOWERCASE = "change_case:english_lowercase"
NO_COMMA = "punctuation:no_comma"
NUMBER_WORDS = "length_constraints:number_words"
JSON_FORMAT = "detectable_format:json_format"
END_CHECKER = "startend:end_checker"
HIGHLIGHTS = "detectable_format:number_highlighted_sections"


# --- strict and loose -----------------------------------------------------------------------


def test_loose_variants_are_the_eight_of_the_released_code_in_order():
    assert loose_variants("*a*\nb\n*c*") == [
        "*a*\nb\n*c*",      # as it is
        "a\nb\nc",          # without '*'
        "b\n*c*",           # without the first line
        "*a*\nb",           # without the last line
        "b",                # without both
        "b\nc",             # without the first line, without '*'
        "a\nb",             # without the last line, without '*'
        "b",                # without both, without '*'
    ]


def test_lowercase_strict_and_loose():
    assert check_instruction(LOWERCASE, {}, "p", "all of this text is lowercase and written in english") == (True, True)
    assert check_instruction(LOWERCASE, {}, "p", "This Has Capitals In It, Written In English") == (False, False)
    # a capitalised first line fails the strict check; the loose check drops it
    assert check_instruction(LOWERCASE, {}, "p", "Sure Thing\nall of this text is lowercase and written in english") \
        == (False, True)


def test_no_comma_loose_drops_a_chatty_first_line():
    assert check_instruction(NO_COMMA, {}, "p", "Sure, here it is:\nno commas in the body") == (False, True)
    assert check_instruction(NO_COMMA, {}, "p", "no commas in the body\nbut, one in the last line, too") == (False, True)
    assert check_instruction(NO_COMMA, {}, "p", "a, b\nc, d\ne, f") == (False, False)


def test_number_words_counts_words():
    kw = {"relation": "at least", "num_words": 5}
    assert check_instruction(NUMBER_WORDS, kw, "p", "one two three four five") == (True, True)
    assert check_instruction(NUMBER_WORDS, kw, "p", "one two three four") == (False, False)
    assert check_instruction(NUMBER_WORDS, {"relation": "less than", "num_words": 5}, "p", "one two") == (True, True)


def test_json_format_loose_drops_the_lead_in_line():
    assert check_instruction(JSON_FORMAT, {}, "p", '{"a": 1, "b": [1, 2]}') == (True, True)
    assert check_instruction(JSON_FORMAT, {}, "p", '```json\n{"a": 1}\n```') == (True, True)
    assert check_instruction(JSON_FORMAT, {}, "p", 'Here you go:\n{"a": 1}') == (False, True)
    assert check_instruction(JSON_FORMAT, {}, "p", "not json at all") == (False, False)


def test_loose_removes_asterisks():
    kw = {"end_phrase": "Is there anything else I can help with?"}
    assert check_instruction(END_CHECKER, kw, "p", "Done. Is there anything else I can help with?") == (True, True)
    assert check_instruction(END_CHECKER, kw, "p", "Done. *Is there anything else I can help with?*") == (False, True)


def test_empty_response_follows_nothing():
    for response in ("", "  \n "):
        assert check_instruction(NO_COMMA, {}, "p", response) == (False, False)


def test_none_arguments_are_dropped():
    """The datasets store every possible argument for every instruction; unused ones are None."""
    kwargs = {"N": None, "num_highlights": 2, "relation": None, "keyword": None, "language": None}
    assert check_instruction(HIGHLIGHTS, kwargs, "p", "*one* and *two*") == (True, True)
    assert check_instruction(HIGHLIGHTS, kwargs, "p", "*one* only") == (False, False)


# --- the metric -----------------------------------------------------------------------------


def _scores(rows):
    """rows: (sample id, epoch, strict list, loose list, truncated) -> one SampleScore each."""
    return [SampleScore(score=Score(value=float(all(s)), metadata={
        "strict": s, "loose": lo, "truncated": t, "epoch": e}), sample_id=i) for i, e, s, lo, t in rows]


def test_metric_pools_every_sample_and_epoch():
    """Sample A has 1 instruction, sample B has 3, two epochs each.

    Instruction-level numbers are micro averages over all 8 (sample, epoch, instruction) cells:
    averaging per-sample fractions instead would give 0.5833 for strict (A: 1/2, B: 2/3)."""
    out = instruction_accuracy()(_scores([
        ("A", 1, [True], [True], False),
        ("A", 2, [False], [True], False),
        ("B", 1, [True, True, True], [True, True, True], False),
        ("B", 2, [True, False, False], [True, True, False], False),
    ]))
    assert out["prompt_level_strict"] == pytest.approx(2 / 4)       # A1, B1
    assert out["prompt_level_loose"] == pytest.approx(3 / 4)        # A1, A2, B1
    assert out["instruction_level_strict"] == pytest.approx(5 / 8)  # 1 + 0 + 3 + 1
    assert out["instruction_level_loose"] == pytest.approx(7 / 8)   # 1 + 1 + 3 + 2
    assert out["truncated"] == 0.0
    assert out["prompt_level_strict_excl_truncated"] == pytest.approx(2 / 4)


def test_metric_reports_truncation():
    out = instruction_accuracy()(_scores([
        ("A", 1, [True], [True], False),
        ("A", 2, [False], [False], True),            # cut off: follows nothing
        ("B", 1, [False, False], [True, False], False),
        ("B", 2, [False, False], [False, False], True),
    ]))
    assert out["truncated"] == pytest.approx(2 / 4)
    assert out["prompt_level_strict"] == pytest.approx(1 / 4)
    assert out["prompt_level_strict_excl_truncated"] == pytest.approx(1 / 2)
    assert out["prompt_level_loose_excl_truncated"] == pytest.approx(1 / 2)


def test_metric_without_scores_is_nan():
    out = instruction_accuracy()([])
    assert all(v != v for k, v in out.items())


# --- the scorer, end to end on mockllm ------------------------------------------------------


@pytest.fixture
def no_nltk_download(monkeypatch):
    """The checks below need no NLTK data; keep the tests from fetching any."""
    monkeypatch.setattr(scorer_module, "_prepare", lambda: None)


def _run(tmp_path, samples, outputs, epochs=1):
    model = get_model("mockllm/model", custom_outputs=itertools.cycle(outputs))
    task = Task(dataset=samples, solver=generate(), scorer=instruction_following(), epochs=epochs)
    return inspect_eval(task, model=model, display="none", log_dir=str(tmp_path / "logs"))[0]


def _metrics(log):
    return {name: m.value for name, m in log.results.scores[0].metrics.items()}


def _sample(i, ids, kwargs):
    return Sample(id=i, input=f"prompt {i}", target="", metadata={"instruction_id_list": ids, "kwargs": kwargs})


def test_scorer_reports_the_four_numbers(tmp_path, no_nltk_download):
    samples = [
        _sample(1, [LOWERCASE], [{}]),
        _sample(2, [NO_COMMA, NUMBER_WORDS], [{}, {"relation": "at least", "num_words": 3, "N": None}]),
    ]
    response = "first, line\nall lowercase words here and more"
    log = _run(tmp_path, samples, [ModelOutput.from_content("mockllm/model", response)])
    m = _metrics(log)
    assert m["prompt_level_strict"] == pytest.approx(1 / 2)             # sample 2: the comma in line 1
    assert m["prompt_level_loose"] == pytest.approx(1.0)                # dropping line 1 fixes it
    assert m["instruction_level_strict"] == pytest.approx(2 / 3)
    assert m["instruction_level_loose"] == pytest.approx(1.0)
    by_id = {s.id: s for s in log.samples}
    score = next(iter(by_id[2].scores.values()))
    assert score.value == 0.0
    assert score.metadata["strict"] == [False, True] and score.metadata["loose"] == [True, True]
    assert score.metadata["instruction_ids"] == [NO_COMMA, NUMBER_WORDS]
    assert score.metadata["truncated"] is False and score.metadata["epoch"] == 1


def test_scorer_counts_every_epoch(tmp_path, no_nltk_download):
    """The same prompt answered well in one epoch and badly in the other counts as 1 of 2."""
    outputs = [ModelOutput.from_content("mockllm/model", t) for t in ("no commas here", "a, b")]
    log = _run(tmp_path, [_sample(1, [NO_COMMA], [{}])], outputs, epochs=2)
    m = _metrics(log)
    assert m["prompt_level_strict"] == pytest.approx(1 / 2) and m["instruction_level_strict"] == pytest.approx(1 / 2)


def test_truncated_response_follows_nothing(tmp_path, no_nltk_download):
    out = ModelOutput.from_content("mockllm/model", "no commas here")
    out.choices[0].stop_reason = "max_tokens"
    log = _run(tmp_path, [_sample(1, [NO_COMMA], [{}])], [out])
    m = _metrics(log)
    assert m["prompt_level_strict"] == 0.0 and m["prompt_level_loose"] == 0.0 and m["truncated"] == 1.0
    score = next(iter(log.samples[0].scores.values()))
    assert score.metadata["truncated"] is True and score.metadata["strict"] == [False]


def test_verifier_error_counts_as_not_followed(tmp_path, no_nltk_download):
    """A malformed argument makes the verifier raise; the sample is scored 0 with the error kept."""
    sample = _sample(1, [NUMBER_WORDS], [{"relation": "bogus", "num_words": 3}])
    log = _run(tmp_path, [sample], [ModelOutput.from_content("mockllm/model", "one two three")])
    score = next(iter(log.samples[0].scores.values()))
    assert score.value == 0.0 and score.metadata["errors"]


# --- NLTK data ------------------------------------------------------------------------------


def test_missing_nltk_data_gives_an_actionable_error(monkeypatch):
    import nltk

    monkeypatch.setattr(scorer_module, "_ready", False)
    monkeypatch.setattr(scorer_module.instructions_util, "download_nltk_resources", lambda: None)

    def not_found(path):
        raise LookupError(path)

    monkeypatch.setattr(nltk.data, "find", not_found)
    with pytest.raises(RuntimeError, match="nltk.downloader.*punkt_tab"):
        scorer_module._prepare()


@pytest.mark.network
def test_nltk_data_can_be_prepared():
    """Downloads punkt_tab, stopwords and the tagger when they are not installed yet."""
    scorer_module._prepare()
    import nltk

    nltk.data.find("tokenizers/punkt_tab")
