"""Read Inspect ``ModelOutput`` objects: reasoning text and the content as the server sent it."""
from __future__ import annotations

from inspect_ai.model import ContentReasoning, ModelOutput


def reasoning_text(output: ModelOutput) -> str:
    """The reasoning in ``output`` (from a reasoning field or a ``<think>`` block), stripped;
    ``""`` if there is none."""
    if not output.choices or isinstance(output.message.content, str):
        return ""
    return "\n".join(c.reasoning for c in output.message.content
                     if isinstance(c, ContentReasoning) and c.reasoning).strip()


def has_reasoning(output: ModelOutput) -> bool:
    """Whether ``output`` carries any reasoning, judged on the raw text (whitespace counts)."""
    if not output.choices or isinstance(output.message.content, str):
        return False
    return any(isinstance(c, ContentReasoning) and c.reasoning for c in output.message.content)


def content_as_sent(output: ModelOutput) -> str:
    """The message ``content`` as the server sent it.

    Inspect moves a ``<think>...</think>`` block out of the text into a ``ContentReasoning``
    (``internal="think"``); this puts it back. Reasoning that came in a separate field
    (``reasoning_content``) was never part of the content and is left out.
    """
    text = output.completion
    if not output.choices or isinstance(output.message.content, str) or text.lstrip().startswith("<think"):
        return text
    think = "".join(f"<think>{c.reasoning}</think>" for c in output.message.content
                    if isinstance(c, ContentReasoning) and c.internal == "think")
    return think + text
