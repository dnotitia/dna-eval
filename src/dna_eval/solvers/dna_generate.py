"""``dna_generate``: call the model under evaluation, retrying empty or failed calls."""
from __future__ import annotations

import asyncio

from inspect_ai.model import ContentReasoning
from inspect_ai.solver import Generate, Solver, TaskState, solver

from dna_eval.generation import MAX_RETRIES

# What the judge is shown when the model gave no answer.
EMPTY_ANSWER = "[모델의 응답이 비어 있음]"


@solver
def dna_generate(attempts: int = MAX_RETRIES) -> Solver:
    """``generate()``, plus the suite's handling of empty and failed responses.

    Every sample is scored, and the judge sees the answer text only:

    - a call that raises, or returns neither content nor reasoning, is retried up to
      ``attempts`` times in total, 1 s then 2 s apart;
    - the answer is the content with surrounding whitespace stripped;
    - no answer (all attempts failed, or only reasoning came back) becomes ``EMPTY_ANSWER``,
      which the judge scores like any other answer instead of the sample erroring out.

    Only ``state.output.completion`` is rewritten; the assistant message (reasoning included)
    is left as the model returned it.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        messages = list(state.messages)
        for attempt in range(1, attempts + 1):
            state.messages = list(messages)
            try:
                state = await generate(state)
                if state.output.completion or _has_reasoning(state):
                    break
            except Exception:  # noqa: BLE001  (any failed call is retried)
                pass
            if attempt < attempts:
                await asyncio.sleep(2 ** (attempt - 1))
        state.output.completion = state.output.completion.strip() or EMPTY_ANSWER
        return state

    return solve


def _has_reasoning(state: TaskState) -> bool:
    if not state.output.choices or isinstance(state.output.message.content, str):
        return False
    return any(isinstance(c, ContentReasoning) and c.reasoning for c in state.output.message.content)
