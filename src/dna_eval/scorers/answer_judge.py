"""``answer_judge``: KMMMU's scoring, a math-verify parse of the answer and a true/false judge.

The answer (the final content) is parsed with ``math_verify.parse``; the ``grader`` model then
sees the question, the gold answer, the answer and its parse, and replies ``true`` or
``false``. Prompts, temperature and the reading of the reply follow HAE-RAE/KMMMU
(``kmmmu_utils.py``, ``eval_tutorial.ipynb``).
"""
from __future__ import annotations

import threading

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.judge_failures import JUDGE_ERROR, judge_failures
from dna_eval.utils.grader import call_grader
from dna_eval.utils.math_answer import last_boxed
from dna_eval.utils.model_output import is_truncated
from dna_eval.utils.optional import require

MATH_INSTALL = 'pip install -e ".[math]"'

# github.com/HAE-RAE/KMMMU (kmmmu_utils.py, eval_tutorial.ipynb), which states no licence;
# see THIRD_PARTY_NOTICES.md.
JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator.\n"
    "Decide if the model's answer matches the gold answer.\n"
    "Use the parsed model answer (from Math-Verify) if helpful.\n"
    "Return ONLY: true or false."
)
JUDGE_TEMPERATURE = 1.0
PARSED = "parsed"              # Score.metadata key: the math-verify parse shown to the judge


def build_prompt(question: str, gold: str, answer: str, parsed: str) -> str:
    """The judge's user turn, as the tutorial builds it."""
    return (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Model output:\n{answer}\n\n"
        f"Parsed model output (Math-Verify parse):\n{parsed}\n\n"
        "Is the model answer correct? Reply only with true or false."
    )


def math_verify_parse(answer: str) -> str:
    """``str(math_verify.parse(answer))``, the tutorial's parse. ``math-verify`` times out with
    signals, which only work on the main thread, so elsewhere it runs without a timeout."""
    from math_verify import parse   # imported here: the package is only needed by the math tasks
    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    try:
        return str(parse(answer, parsing_timeout=timeout))
    except Exception:  # noqa: BLE001  (a parse that fails or times out shows as empty, as math-verify returns)
        return "[]"


def read_verdict(reply: str) -> bool:
    """Correct when the reply, stripped and lower-cased, starts with ``true``."""
    return reply.strip().lower().startswith("true")


@scorer(metrics=[accuracy(), stderr(), answer_diagnostics(), judge_failures()])
def answer_judge(question_key: str = "question", judge_retries: int = 2) -> Scorer:
    """Score 1.0 when the grader says the final content matches the target, else 0.0.

    - A truncated or empty answer scores 0 without asking the grader.
    - ``EXTRACTED`` is the last ``\\boxed{}`` of the answer (the format the system prompt asks
      for), so ``answer_diagnostics`` counts answers without one; the judge sees the whole answer.
    - A failed grader call is retried ``judge_retries`` times, then scores 0 and is recorded as
      ``JUDGE_ERROR = "call"``; any reply that does not start with ``true`` is wrong, as in
      the reference.
    """
    require("math_verify", install=MATH_INSTALL)

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion.strip()
        truncated = is_truncated(state.output)
        md: dict = {TRUNCATED: truncated, EXTRACTED: last_boxed(answer)}
        if truncated or not answer:
            return Score(value=0.0, answer=answer, metadata=md,
                         explanation="truncated" if truncated else "empty answer")
        md[PARSED] = math_verify_parse(answer)
        prompt = build_prompt(state.metadata[question_key], target.text, answer, md[PARSED])
        reply, error = await call_grader([ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT), ChatMessageUser(content=prompt)],
                                         GenerateConfig(temperature=JUDGE_TEMPERATURE), judge_retries)
        if error:
            md[JUDGE_ERROR] = error
        correct = not error and read_verdict(reply)
        return Score(value=1.0 if correct else 0.0, answer=answer, metadata=md,
                     explanation=(f"judge {error} failure" if error else f"grader replied {reply[:100]!r}"))

    return score
