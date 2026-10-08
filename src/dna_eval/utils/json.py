"""Pull a JSON object out of free-form model output."""
from __future__ import annotations

import json
import re

_TRAILING_COMMA = re.compile(r",\s*([\]}])")


def extract_json(text: str) -> dict:
    """Parse the span from the first ``{`` to the last ``}`` in ``text``.

    A port of DeepEval's ``trim_and_load_json`` (deepeval 4.0.3), kept identical so judge
    replies are read the same way: a missing closing brace is appended, trailing commas
    are dropped, and anything else unparsable raises ``ValueError``.
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if end == 0 and start != -1:
        text += "}"
        end = len(text)
    span = text[start:end] if start != -1 and end != 0 else ""
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", span))
    except json.JSONDecodeError as e:
        raise ValueError(f"no JSON object in text: {e}") from e
