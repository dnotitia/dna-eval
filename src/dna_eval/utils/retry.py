"""Retry a model call until it returns something."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from inspect_ai.model import ModelOutput
from inspect_ai.util import LimitExceededError

from dna_eval.utils.model_output import has_reasoning


async def call_with_retries(call: Callable[[], Awaitable[ModelOutput]], attempts: int) -> ModelOutput | None:
    """Run ``call`` until it returns content or reasoning, at most ``attempts`` times, 1 s, 2 s,
    4 s ... apart. Errors count as failed attempts. Returns the last output, or ``None`` if
    every attempt raised.

    An Inspect ``LimitExceededError`` (a sample limit, such as a token or message limit, reached
    during a call) is not a failed attempt: Inspect refuses every further call before sending
    it, so retrying would only wait. The error propagates with ``last_output`` set to what the
    retries would have returned (the last output a call returned, or ``None``)."""
    output = None
    for attempt in range(1, attempts + 1):
        try:
            output = await call()
            if output.completion or has_reasoning(output):
                return output
        except LimitExceededError as ex:
            ex.last_output = output
            raise
        except Exception:  # noqa: BLE001  (any failed call is retried)
            pass
        if attempt < attempts:
            await asyncio.sleep(2 ** (attempt - 1))
    return output
