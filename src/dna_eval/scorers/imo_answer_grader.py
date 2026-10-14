"""``imo_answer_grader``: IMO-AnswerBench correctness, graded by the paper's AnswerAutoGrader prompt.

The prompt (``GRADER_PROMPT``) is Appendix A.5 of "Towards Robust Mathematical
Reasoning" (arXiv 2511.01846), except for the "# 2. Extraction Protocol" section the paper
refers to but does not print: that section is this project's own, and says the final answer is
the last ``\\boxed{}`` of the solution. The grader is the model bound to the ``grader`` role
(required) and sees the problem, the reference answer and the subject's final content, never its
reasoning.
"""
from __future__ import annotations

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.judge_failures import JUDGE_ERROR, judge_failures
from dna_eval.roles import GRADER_ROLE
from dna_eval.utils.math_answer import last_boxed
from dna_eval.utils.model_output import is_truncated
from dna_eval.utils.templates import render

# Grader prompt for IMO-AnswerBench: "AnswerAutoGrader", Appendix A.5 of "Towards Robust
# Mathematical Reasoning" (Luong et al., 2025, arXiv 2511.01846; CC-BY-4.0, see
# THIRD_PARTY_NOTICES.md), as a Jinja template. Reproduced from the paper's listing, with its
# wording and typo ("Output exmaple") kept; only the listing's line wraps are joined and the
# paper's typeset quotes around code spans are written as backticks.
# NOT IN THE PAPER: section "# 2. Extraction Protocol". Section 3 refers to an "Extraction
# Protocol", but the published prompt jumps from section 1 to section 3; the section 2 here is
# this project's minimal stand-in.
GRADER_PROMPT = r"""# System Role: Deterministic Mathematical Autograder
You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer. You must NOT grade the reasoning or steps, only the final result.
# 1. Grading Guidelines (Equivalence Rules)
Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.
* **Algebraic Equivalence:** e.g., `n(n+1)/2` is equivalent to `n^2/2 + n/2`. You must verify the algebra.
* **Numerical Equivalence:** e.g., `1/2` is equivalent to `0.5`; `sqrt(2)/2` is equivalent to `1/sqrt(2)`.
* **Set/List Equivalence:** Unless specified as an ordered tuple/vector, the order of elements does not matter (e.g., {1, 2} is equivalent to {2, 1}).
* **Partial Credit:** No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
* **No Answers:** If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.
# 2. Extraction Protocol
The final answer is the one the Model Solution commits to: the content of the last `\boxed{}` expression in the Model Solution. If the Model Solution contains no `\boxed{}` expression, it has no final answer.
# 3. Output Protocol (Strict Compliance Required)
You must execute the task using a two-part structure. Failure to follow this structure will result in task failure.
**Part 1: Analysis (Chain-of-Thought)**
You MUST perform your analysis within <thinking></thinking> tags. Make your thinking concise. This section details your reasoning process and must follow these steps sequentially:
1. **Golden Answer:** State the Golden Answer.
2. **Extracted Model Answer:** State the extracted answer based on the Extraction Protocol. If none found, state "No clear final answer found."
3. **Equivalence Analysis:** Compare the two answers using the Grading Guidelines. Detail the steps taken to verify mathematical equivalence (e.g., simplification, algebraic manipulation). You must actively try to prove they are the same before concluding they are different.
4. **Conclusion:** State the final determination ("Correct" or "Incorrect").
**Part 2: Final Grade**
Immediately following the closing </thinking> tag, output **ONLY** the final grade.
* If Correct: \boxed{Correct}
* If Incorrect: \boxed{Incorrect}
**CRITICAL CONSTRAINT: Do not add any text, explanations, or formatting outside the <thinking> tags or the final \boxed{} output.**

Output exmaple:
<thinking>
1. **Golden Answer:** (−∞, −4) ∪ (−4, ∞)
2. **Extracted Model Answer:** ∅ (the empty set)
3. **Equivalence Analysis:**
The Golden Answer is a non-empty set of real numbers. The Model Answer is the empty set. These two sets are not equivalent. The empty set contains no elements, while the Golden Answer contains an infinite number of elements.
4. **Conclusion:** Incorrect
</thinking>
\boxed{Incorrect}

# 4. Input Data
Here is the problem, model solution, and golden answer to grade:
Problem: {{ problem }}
Model Solution: {{ solution }}
Golden Answer: {{ golden_answer }}"""

_VERDICT = re.compile(r"\\boxed\{\s*(Correct|Incorrect)\s*\}")


def read_grade(reply: str) -> bool:
    """Whether the grader said ``Correct``: the last ``\\boxed{Correct}`` / ``\\boxed{Incorrect}``
    in ``reply`` decides. Raises ``ValueError`` if the reply holds neither."""
    found = _VERDICT.findall(reply)
    if not found:
        raise ValueError("no \\boxed{Correct} or \\boxed{Incorrect} in the grader reply")
    return found[-1] == "Correct"


@scorer(metrics=[accuracy(), stderr(), answer_diagnostics(), judge_failures()])
def imo_answer_grader(judge_retries: int = 2) -> Scorer:
    """Score 1 when the grader's last ``\\boxed{}`` says ``Correct``, else 0.

    A grader reply with no ``\\boxed{Correct}`` / ``\\boxed{Incorrect}``, or a failed call, is
    repeated up to ``judge_retries`` more times and then scores 0. A subject answer that was cut
    off scores 0 without a grader call. The grader is called with no temperature or token
    limit set (the paper gives none).

    The problem the grader reads is the sample's ``metadata["problem"]`` (the statement alone,
    without the instruction added to the subject's prompt).

    ``Score.metadata``: ``extracted`` (the last ``\\boxed{}`` of the subject's answer, read here
    for the diagnostics; ``None`` when there is none, so ``format_failure`` counts answers
    without one), ``truncated``, ``judge_error`` (``"call"`` or ``"parse"``, when no verdict was obtained).
    """

    async def score(state: TaskState, target: Target) -> Score:
        solution = state.output.completion
        md: dict = {TRUNCATED: is_truncated(state.output), EXTRACTED: last_boxed(solution)}
        if md[TRUNCATED]:
            return Score(value=0.0, answer=md[EXTRACTED] or "", explanation="answer cut off", metadata=md)
        prompt = render(GRADER_PROMPT,
            problem=state.metadata["problem"], solution=solution, golden_answer=target.text)
        error = ""
        for _ in range(1 + judge_retries):
            grader = get_model(role=GRADER_ROLE, required=True)    # a missing role stops the eval
            try:
                reply = (await grader.generate(prompt)).completion
            except Exception as e:  # noqa: BLE001  (any failed call is retried)
                error = f"call: {type(e).__name__}: {e}"[:300]
                continue
            try:
                correct = read_grade(reply)
            except ValueError:
                error = f"parse: {reply[-300:]}"
                continue
            return Score(value=1.0 if correct else 0.0, answer=md[EXTRACTED] or "", explanation=reply, metadata=md)
        return Score(value=0.0, answer=md[EXTRACTED] or "", explanation=f"grader failed: {error}",
                     metadata=md | {JUDGE_ERROR: error.split(":", 1)[0]})

    return score
