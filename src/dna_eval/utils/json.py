"""Pull a JSON object out of free-form model output.

``extract_json`` is ported from DeepEval (confident-ai/deepeval, Apache-2.0, Copyright 2024
Confident AI Inc.); see NOTICE.
"""
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


# A backslash that starts a valid JSON escape (kept as it is), or any other backslash.
_ESCAPE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})|\\')


def escape_stray_backslashes(text: str) -> str:
    """``text`` with every backslash that does not start a valid JSON escape doubled.

    Models quoting LaTeX inside a JSON string write ``\\pi`` or ``\\mid`` with one backslash,
    which ``json.loads`` rejects. Valid escapes (``\\n``, ``\\"``, ``\\\\``, ``\\uXXXX``, ...)
    are left alone, so text that already parses is unchanged.
    """
    return _ESCAPE.sub(lambda m: m.group(0) if m.group(1) else "\\\\", text)
