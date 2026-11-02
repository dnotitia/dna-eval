"""``letter_choice``: score a multiple-choice answer given as a letter in the model's reply.

The letter is read from ``state.output.completion``, the final content. Reasoning is never
searched, so an "Answer: X" written while thinking does not count. Matching is configurable
(regular expressions, the accepted letters, text to remove first) and always takes the *last*
match, because a reasoning model may mention an answer line before it commits to one.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.utils.model_output import is_truncated

# Further ``Score.metadata`` keys.
EXPECTED = "expected"          # the correct letter for the sample
CANDIDATES = "candidates"      # distinct letters named in the reply (``unique_letter`` mode)


def extract_letter(text: str, patterns: Sequence[re.Pattern[str]], letters: str,
                   unique_letter: bool = False) -> tuple[str | None, list[str]]:
    """The answer letter in ``text`` and the distinct candidate letters found.

    Default mode: patterns are tried in order; the first one whose *last* match captures a
    letter in ``letters`` decides the answer.

    ``unique_letter`` mode: every match of every pattern is an answer statement; the letters
    from ``letters`` inside the captured text of all statements are collected, and the answer
    is the letter only if exactly one distinct letter was named. Naming several letters, or
    none, gives no answer.
    """
    if unique_letter:
        found: list[str] = []
        for pattern in patterns:
            for statement in pattern.findall(text):
                found.extend(c for c in statement if c in letters and c not in found)
        return (found[0] if len(found) == 1 else None), found
    for pattern in patterns:
        matches = pattern.findall(text)
        if matches and matches[-1] in letters:
            return matches[-1], [matches[-1]]
    return None, []


@scorer(metrics=[accuracy(), stderr(), answer_diagnostics()])
def letter_choice(patterns: Sequence[str], letters: str = "ABCD", unique_letter: bool = False,
                  strip: Sequence[str] = (), target_key: str | None = None) -> Scorer:
    """Score 1.0 if the letter read from the final content is the correct one, else 0.0.

    ``patterns`` are regular expressions with one capture group (inline flags such as
    ``(?i)`` allowed), tried in order; the *last* match of the first pattern that yields a
    letter in ``letters`` is the answer (letters are compared case-sensitively). ``strip``
    lists substrings removed from the reply before matching (e.g. markdown bold markers).

    With ``unique_letter`` the reply must name exactly one distinct letter across all its
    answer statements (see ``extract_letter``); a reply that names two is wrong.

    The correct letter is the sample's target, or ``state.metadata[target_key]`` when the
    task decides it per run (shuffled options). A truncated reply scores 0 even if it
    contains an answer line. ``Score.answer`` and ``metadata`` hold ``TRUNCATED``,
    ``EXTRACTED`` (the letter, or ``None``), ``EXPECTED`` and, in ``unique_letter`` mode,
    ``CANDIDATES``.
    """
    compiled = [re.compile(p) for p in patterns]
    for p in compiled:
        if p.groups != 1:
            raise ValueError(f"letter_choice patterns need exactly one capture group: {p.pattern!r}")

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        for s in strip:
            text = text.replace(s, "")
        extracted, candidates = extract_letter(text, compiled, letters, unique_letter)
        expected = str(state.metadata[target_key] if target_key else target.text).strip()
        truncated = is_truncated(state.output)
        correct = extracted is not None and extracted == expected and not truncated
        metadata = {TRUNCATED: truncated, EXTRACTED: extracted, EXPECTED: expected}
        if unique_letter:
            metadata[CANDIDATES] = candidates
        return Score(value=1.0 if correct else 0.0, answer=extracted, metadata=metadata,
                     explanation=f"extracted {extracted!r}, expected {expected!r}"
                                 + (", truncated" if truncated else ""))

    return score
