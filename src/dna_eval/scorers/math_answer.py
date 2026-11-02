"""``math_answer``: score a final answer to a math problem against a reference answer."""
from __future__ import annotations

from typing import Literal

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.utils.math_answer import (extract_boxed_answer, is_equiv_math_verify, is_equiv_string,
                                        last_boxed, last_integer, parse_integer)
from dna_eval.utils.model_output import is_truncated
from dna_eval.utils.optional import require

MATH_INSTALL = 'pip install "dna-eval[math]"'   # math-verify, for mode="expression"


@scorer(metrics=[accuracy(), stderr(), answer_diagnostics()])
def math_answer(mode: Literal["integer", "expression"] = "expression") -> Scorer:
    """Compare the answer in ``state.output.completion`` (the final content, never the
    reasoning) with the sample's target. Scores 1.0 or 0.0.

    - ``mode="integer"`` (competition math with whole-number answers): the answer is the
      content of the last ``\\boxed{}``/``\\fbox{}``; with no box, the last integer in the
      response. It is correct when it reads as the same integer as the target, so ``042``
      and ``42.0`` both match 42.
    - ``mode="expression"`` (answers that may be numbers, fractions, radicals, sets, ...): the
      answer is read as the lm-evaluation-harness HRM8K task reads it (last box, the text after
      its last ``=``, with fallbacks for a response without a box). It is correct when it
      equals the target under the Hendrycks et al. string normalisation, or when ``math-verify``
      finds the two mathematically equal.

    A truncated response scores 0 whatever it contains. ``metadata`` holds ``truncated``,
    ``extracted`` (the answer that was compared, ``None`` if none was found) and ``boxed``
    (whether the response contained a box).
    """
    if mode not in ("integer", "expression"):
        raise ValueError(f"mode must be 'integer' or 'expression', got {mode!r}")

    if mode == "expression":
        require("math_verify", install=MATH_INSTALL)

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        truncated = is_truncated(state.output)
        boxed = last_boxed(text) is not None
        if mode == "integer":
            extracted = last_boxed(text)
            if extracted is None:
                n = last_integer(text)
                extracted = None if n is None else str(n)
            gold = parse_integer(target.text)
            pred = None if extracted is None else parse_integer(extracted)
            correct = gold is not None and pred is not None and pred == gold
            how = "integer"
        else:
            extracted = extract_boxed_answer(text)
            by_string = is_equiv_string(extracted, target.text)
            by_symbols = not by_string and is_equiv_math_verify(extracted, target.text)
            correct = by_string or by_symbols
            how = "string" if by_string else "math-verify" if by_symbols else "none"
        correct = correct and not truncated
        reason = ("truncated" if truncated else "no answer found" if extracted is None
                  else f"matched ({how})" if correct else "wrong")
        return Score(value=1.0 if correct else 0.0, answer=extracted, explanation=reason,
                     metadata={TRUNCATED: truncated, EXTRACTED: extracted, "boxed": boxed})

    return score
