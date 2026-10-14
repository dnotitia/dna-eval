"""``hle_judge``: Humanity's Last Exam correctness, judged as the benchmark's own script does.

The judge prompt (``JUDGE_PROMPT``) is the one in ``hle_eval/run_judge_results.py`` of
github.com/centerforaisafety/hle, typos included. The judge extracts the response's final answer,
says whether it matches the reference (``correct``: ``yes``/``no``) and extracts the confidence
the response stated (``confidence``: 0-100). The judge is the model bound to the ``grader`` role
(required); it sees the question, the reference and the subject's final content, never its
reasoning.
"""
from __future__ import annotations

import re
from typing import Literal

from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, field_validator

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.calibration_error import CONFIDENCE, calibration_error
from dna_eval.metrics.judge_failures import JUDGE_ERROR, judge_failures
from dna_eval.roles import GRADER_ROLE
from dna_eval.utils.json import escape_stray_backslashes, extract_json
from dna_eval.utils.model_output import is_truncated
from dna_eval.utils.templates import render

# The judge prompt of Humanity's Last Exam, verbatim from hle_eval/run_judge_results.py (typos
# included), as a Jinja template. Source: centerforaisafety/hle (MIT, Copyright (c) 2025
# centerforaisafety); see THIRD_PARTY_NOTICES.md.
JUDGE_PROMPT = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {{ question }}

[response]: {{ response }}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {{ correct_answer }}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class Judgement(BaseModel):
    """The four fields the judge fills in, as in the official ``ExtractedAnswer`` model."""
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int

    @field_validator("correct", mode="before")
    @classmethod
    def _normalise_correct(cls, v):
        return v.strip().strip(".'\"*").lower() if isinstance(v, str) else v

    @field_validator("confidence", mode="before")
    @classmethod
    def _strip_percent(cls, v):
        return v.strip().rstrip("%").strip() if isinstance(v, str) else v


_FIELD = re.compile(r"^[\s*_#>-]*(extracted_final_answer|reasoning|correct|confidence)[\s*_]*:[\s*_]*",
                    re.IGNORECASE | re.MULTILINE)


def _labelled_fields(reply: str) -> dict[str, str]:
    """``name: value`` blocks in a plain-text reply (the judge prompt lists the fields that way)."""
    found = list(_FIELD.finditer(reply))
    out: dict[str, str] = {}
    for m, nxt in zip(found, found[1:] + [None]):
        out.setdefault(m.group(1).lower(), reply[m.end(): nxt.start() if nxt else len(reply)].strip())
    return out


def read_judgement(reply: str) -> Judgement:
    """The judgement in a judge reply: a JSON object (structured output, or JSON in prose), else
    ``name: value`` lines. A JSON object whose strings quote LaTeX with single backslashes
    (``\\frac{\\pi}{2}``) is read after escaping them. Raises ``ValueError`` when a field is
    missing or malformed."""
    try:
        fields = extract_json(reply)
    except ValueError:
        try:
            fields = extract_json(escape_stray_backslashes(reply))
        except ValueError:
            fields = _labelled_fields(reply)
    try:
        return Judgement.model_validate(fields)
    except ValueError as e:      # pydantic.ValidationError is a ValueError
        raise ValueError(f"unreadable judgement: {e}") from e


def _extracted(answer: str) -> str | None:
    """The judge's extracted answer, or ``None`` when it found none (``'None'`` or empty)."""
    answer = answer.strip()
    return None if answer.lower() in ("", "none") else answer


@scorer(metrics=[accuracy(), stderr(), calibration_error(), answer_diagnostics(), judge_failures()])
def hle_judge(judge_retries: int = 2) -> Scorer:
    """Score 1 when the judge says the response's final answer matches the reference, else 0.

    The judge is asked for the four fields as structured output (``response_schema``). A server
    that ignores it is covered by reading JSON out of the reply, or ``name: value`` lines. A
    judge call that fails or cannot be read is repeated up to ``judge_retries`` more times; if
    the call itself raised, the repeat asks for plain text in case the server rejects
    ``response_schema``. A judgement that never succeeds scores 0 (the official script counts a
    missing judgement as wrong) and is left out of the calibration error.

    A subject answer that was cut off scores 0 without a judge call, and is likewise left out
    of the calibration error. The judge is called with no temperature or token limit set, since
    the official judge is a reasoning model that accepts neither.

    ``Score.metadata``: ``confidence`` (0-100 or ``None``), ``extracted`` (the judge's extracted
    answer or ``None``), ``truncated``, ``judge_error`` (``"call"`` or ``"parse"``, when no judgement was obtained).
    """

    async def score(state: TaskState, target: Target) -> Score:
        md: dict = {TRUNCATED: is_truncated(state.output), EXTRACTED: None, CONFIDENCE: None}
        if md[TRUNCATED]:
            return Score(value=0.0, answer="", explanation="answer cut off", metadata=md)
        prompt = render(JUDGE_PROMPT,
            question=state.user_prompt.text, response=state.output.completion, correct_answer=target.text)
        schema = ResponseSchema(name="extracted_answer", json_schema=json_schema(Judgement))
        use_schema, error = True, ""
        for _ in range(1 + judge_retries):
            config = GenerateConfig(response_schema=schema) if use_schema else GenerateConfig()
            grader = get_model(role=GRADER_ROLE, required=True, config=config)   # a missing role stops the eval
            try:
                reply = (await grader.generate(prompt)).completion
            except Exception as e:  # noqa: BLE001  (any failed call is retried)
                use_schema, error = False, f"call: {type(e).__name__}: {e}"[:300]
                continue
            try:
                j = read_judgement(reply)
            except ValueError:
                error = f"parse: {reply[:300]}"
                continue
            md |= {EXTRACTED: _extracted(j.extracted_final_answer), CONFIDENCE: j.confidence}
            return Score(value=1.0 if j.correct == "yes" else 0.0, answer=j.extracted_final_answer,
                         explanation=j.reasoning, metadata=md)
        return Score(value=0.0, answer="", explanation=f"judge failed: {error}",
                     metadata=md | {JUDGE_ERROR: error.split(":", 1)[0]})

    return score
