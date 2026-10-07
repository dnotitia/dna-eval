"""Pull a JSON object out of free-form model output."""
from __future__ import annotations

import json
import re

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict:
    """Return the first ``{...}`` block in ``text`` as a dict; tolerates code fences and prose."""
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError("no JSON object in text")
    return json.loads(m.group(0))
