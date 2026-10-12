"""``dna_generate``: call the model under evaluation, retrying empty or failed calls."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from inspect_ai.model import ContentReasoning, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

# What the judge is shown when the model gave no answer, or no reasoning.
EMPTY_ANSWER = "[모델의 응답이 비어 있음]"
EMPTY_REASONING = "[추론 없음]"

# ``state.store`` key holding the reasoning text (see ``rubrics``: ``sources: {...: reasoning}``).
REASONING_KEY = "reasoning"

ATTEMPTS = 3


@solver
def dna_generate(attempts: int = ATTEMPTS) -> Solver:
    """``generate()``, plus the suite's handling of empty and failed responses.

    Every sample is scored, and the judge sees the answer text only:

    - a call that raises, or returns neither content nor reasoning, is retried up to
      ``attempts`` times in total, 1 s then 2 s apart;
    - the answer is the content with surrounding whitespace stripped;
    - no answer (all attempts failed, or only reasoning came back) becomes ``EMPTY_ANSWER``,
      which the judge scores like any other answer instead of the sample erroring out;
    - the reasoning (from a ``reasoning_content`` / ``reasoning`` field or a ``<think>`` block),
      stripped, is kept in ``state.store["reasoning"]``, or ``EMPTY_REASONING`` if there is none.

    Only ``state.output.completion`` is rewritten; the assistant message (reasoning included)
    is left as the model returned it.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        messages = list(state.messages)

        async def call() -> ModelOutput:
            nonlocal state
            state.messages = list(messages)
            state = await generate(state)
            return state.output

        await call_with_retries(call, attempts)
        state.output.completion = state.output.completion.strip() or EMPTY_ANSWER
        state.store.set(REASONING_KEY, reasoning_text(state.output) or EMPTY_REASONING)
        return state

    return solve


async def call_with_retries(call: Callable[[], Awaitable[ModelOutput]], attempts: int = ATTEMPTS) -> ModelOutput | None:
    """Run ``call`` until it returns content or reasoning, at most ``attempts`` times, 1 s then
    2 s apart. Errors count as failed attempts. Returns the last output, or ``None`` if every
    attempt raised."""
    output = None
    for attempt in range(1, attempts + 1):
        try:
            output = await call()
            if output.completion or _has_reasoning(output):
                return output
        except Exception:  # noqa: BLE001  (any failed call is retried)
            pass
        if attempt < attempts:
            await asyncio.sleep(2 ** (attempt - 1))
    return output


def reasoning_text(output: ModelOutput) -> str:
    """The model's reasoning in ``output``, stripped; ``""`` if there is none."""
    if not output.choices or isinstance(output.message.content, str):
        return ""
    return "\n".join(c.reasoning for c in output.message.content
                     if isinstance(c, ContentReasoning) and c.reasoning).strip()


def _has_reasoning(output: ModelOutput) -> bool:
    # Raw, not stripped: a whitespace-only reasoning still counts as a reply, as in the
    # retry rule this mirrors.
    if not output.choices or isinstance(output.message.content, str):
        return False
    return any(isinstance(c, ContentReasoning) and c.reasoning for c in output.message.content)
