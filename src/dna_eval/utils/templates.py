"""Render a Jinja template held in a string."""
from __future__ import annotations

from functools import lru_cache

from jinja2 import Environment, StrictUndefined, Template

_env = Environment(
    undefined=StrictUndefined,       # a typo in a template fails loudly instead of rendering blank
    keep_trailing_newline=True,      # a template's final newline is part of the prompt
    autoescape=False,
)


@lru_cache(maxsize=None)
def _compiled(source: str) -> Template:
    return _env.from_string(source)


def render(source: str, /, **values) -> str:
    """``source`` rendered with ``values``; the text outside Jinja tags comes out byte for byte."""
    return _compiled(source).render(**values)
