"""Call the judge bound to the ``grader`` role, retrying a failed call."""
from __future__ import annotations

import asyncio

from inspect_ai.model import ChatMessage, GenerateConfig, get_model

from dna_eval.roles import GRADER_ROLE


async def call_grader(messages: list[ChatMessage], config: GenerateConfig, retries: int) -> tuple[str, str]:
    """``(reply, error)``: the grader's final content, or ``("", "call")`` when the call still
    fails after ``retries`` retries, 1 s, 2 s, 4 s ... apart. Whether the reply can be read is
    for the caller to decide."""
    grader = get_model(role=GRADER_ROLE, required=True, config=config)
    for attempt in range(1 + retries):
        try:
            return (await grader.generate(messages)).completion, ""
        except Exception:  # noqa: BLE001  (any failed call is retried)
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    return "", "call"
