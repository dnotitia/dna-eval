"""``escape_stray_backslashes`` repairs LaTeX backslashes and leaves valid JSON alone."""
import json

import pytest

from dna_eval.utils.json import escape_stray_backslashes


@pytest.mark.parametrize("raw, reason", [
    (r'{"reason": "($n, \mid$)"}', r"($n, \mid$)"),
    (r'{"reason": "\pi, \le_p, \sum"}', r"\pi, \le_p, \sum"),
    (r'{"reason": "\underline{x}"}', r"\underline{x}"),              # \u without four hex digits
    (r'{"reason": "끝에 \\"}', "끝에 \\"),
])
def test_stray_backslashes_are_escaped(raw, reason):
    assert json.loads(escape_stray_backslashes(raw))["reason"] == reason


@pytest.mark.parametrize("raw", [
    r'{"reason": "줄\n바꿈, 따옴표 \", 역슬래시 \\pi, 유니코드 가, \/"}',
    r'{"reason": "\frac{1}{2}, \beta, \times, \rho"}',               # valid escapes, as JSON reads them
    '{"score": 8, "reason": "ok"}',
])
def test_text_that_parses_is_unchanged(raw):
    assert escape_stray_backslashes(raw) == raw
