"""``imo_answer_grader``: IMO-AnswerBench correctness, graded by the paper's AnswerAutoGrader prompt.

The prompt (``prompts/imo_answer_grader.jinja``) is Appendix A.5 of "Towards Robust Mathematical
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
from dna_eval.prompts import env
from dna_eval.roles import GRADER_ROLE
from dna_eval.utils.math_answer import last_boxed
from dna_eval.utils.model_output import is_truncated

TEMPLATE = "imo_answer_grader.jinja"

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
        prompt = env.get_template(TEMPLATE).render(
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
