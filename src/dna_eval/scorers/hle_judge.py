"""``hle_judge``: Humanity's Last Exam correctness, judged as the benchmark's own script does.

The judge prompt (``prompts/hle_judge.jinja``) is the one in ``hle_eval/run_judge_results.py`` of
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
from inspect_ai.scorer import Score, Scorer, Target, accuracy, grouped, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import json_schema
from pydantic import BaseModel, field_validator

from dna_eval.judging import GRADER_ROLE
from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.calibration_error import CONFIDENCE, calibration_error
from dna_eval.prompts import env
from dna_eval.utils.json import extract_json
from dna_eval.utils.model_output import is_truncated

TEMPLATE = "hle_judge.jinja"


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
    ``name: value`` lines. Raises ``ValueError`` when a field is missing or malformed."""
    try:
        fields = extract_json(reply)
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


@scorer(metrics=[accuracy(), stderr(), grouped(accuracy(), "category", all=False),
                 calibration_error(), answer_diagnostics()])
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
    answer or ``None``), ``truncated``, ``judge_error`` (when no judgement was obtained).
    """

    async def score(state: TaskState, target: Target) -> Score:
        md: dict = {TRUNCATED: is_truncated(state.output), EXTRACTED: None, CONFIDENCE: None}
        if md[TRUNCATED]:
            return Score(value=0.0, answer="", explanation="answer cut off", metadata=md)
        prompt = env.get_template(TEMPLATE).render(
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
        return Score(value=0.0, answer="", explanation=f"judge failed: {error}", metadata=md | {"judge_error": error})

    return score
