"""``shuffled_choices``: build the multiple-choice prompt with the options in a seeded random order."""
from __future__ import annotations

import hashlib
import random

from inspect_ai.solver import Generate, Solver, TaskState, solver

CHOICES_KEY = "choices"          # state.metadata: the options, correct one first
CORRECT_LETTER_KEY = "correct_letter"   # state.metadata: written here, read by the scorer
ORDER_KEY = "choice_order"       # state.metadata: the permutation used (indices into the options)


def permutation(n: int, seed: int, sample_id: object, epoch: int) -> list[int]:
    """A permutation of ``range(n)`` fixed by ``(seed, sample_id, epoch)`` and nothing else.

    Not Python's ``hash``, which differs between processes, so a rerun reproduces the order.
    """
    digest = hashlib.sha256(f"{seed}/{sample_id}/{epoch}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).sample(range(n), n)


@solver
def shuffled_choices(template: str, seed: int = 0, letters: str = "ABCD") -> Solver:
    """Replace the user turn with ``template`` filled in with the options in a shuffled order.

    The sample's input is the question and ``state.metadata["choices"]`` lists its options
    with the correct one first. ``template`` uses ``{Question}`` and one placeholder per
    letter (``{A}``, ``{B}``, ...). The order is a deterministic function of ``seed``, the
    sample id and the epoch: it differs between epochs of the same sample, and a rerun
    reproduces it. The letter of the correct option is stored in
    ``state.metadata["correct_letter"]`` (for the first copy if two options are identical),
    the permutation in ``state.metadata["choice_order"]``.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        choices = list(state.metadata[CHOICES_KEY])
        order = permutation(len(choices), seed, state.sample_id, state.epoch)
        shuffled = [choices[i] for i in order]
        fields = {letter: text for letter, text in zip(letters, shuffled)}
        state.user_prompt.text = template.format(Question=state.input_text, **fields)
        state.metadata[CORRECT_LETTER_KEY] = letters[shuffled.index(choices[0])]
        state.metadata[ORDER_KEY] = order
        return state

    return solve
