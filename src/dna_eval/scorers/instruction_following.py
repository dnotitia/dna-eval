"""``instruction_following``: check a response against the verifiable instructions of its prompt.

Used by IFEval and IFBench. Each sample carries ``metadata["instruction_id_list"]`` (ids such as
``punctuation:no_comma``) and ``metadata["kwargs"]`` (one dict of arguments per instruction).
The verifiers are the vendored IFBench package (``vendor/ifbench``), whose registry holds the
original IFEval instructions as well as IFBench's own.

The response is checked the way the benchmarks' released code (``evaluation_lib.py``) does it:

- **strict**: the response as it is;
- **loose**: an instruction counts as followed if any of eight variants of the response passes:
  the response, and the response without its first line, without its last line, and without
  both, each once as it is and once with every ``*`` removed (so markdown emphasis does not
  break a check).

An empty response follows nothing. Arguments whose value is ``None`` are dropped before the
instruction is built, as the datasets store every possible argument for every instruction.
Instructions that quote the prompt (``combination:repeat_prompt``) get the sample's input.

Deviations from the released code:

- a response cut off at a token or context limit follows nothing, strict or loose
  (decision record 0015), instead of being checked as far as it goes;
- a verifier that raises is counted as not followed and the error is kept in
  ``metadata["errors"]``, instead of stopping the run;
- the language detector is seeded (``langdetect`` is otherwise non-deterministic on short or
  ambiguous text).

Known quirk kept as it is: IFEval's ``keywords:letter_frequency`` verifier replaces a letter
argument that is not a letter (two prompts ask about ``#`` and ``!``) with a random letter and
frequency, so those two prompts are scored against a random constraint.

NLTK data (``punkt_tab``, ``stopwords``, ``averaged_perceptron_tagger_eng``) is needed by some
verifiers. It is downloaded the first time a response is scored, which needs network access
once; afterwards it is read from NLTK's data directory (``~/nltk_data`` or ``NLTK_DATA``).
To fetch it in advance: ``python -m nltk.downloader punkt punkt_tab stopwords
averaged_perceptron_tagger_eng``.

The score value is 1.0 when every instruction passed the strict check, else 0.0.
``metadata`` holds ``strict`` and ``loose`` (a bool per instruction), ``instruction_ids``,
``truncated``, ``extracted`` (the response, or ``None`` if it is empty), ``epoch`` and, if a
verifier raised, ``errors``. ``metrics/instruction_following.py`` turns these into the four
official accuracies.
"""
from __future__ import annotations

import logging

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED
from dna_eval.metrics.instruction_following import LOOSE, STRICT, TRUNCATED, instruction_accuracy
from dna_eval.utils.model_output import is_truncated
from dna_eval.vendor.ifbench import instructions_registry, instructions_util

logger = logging.getLogger(__name__)

# (resource path for nltk.data.find, name for nltk.download) of what the verifiers read.
_NLTK_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)
_ready = False


def _prepare() -> None:
    """Fetch the NLTK data the verifiers need (once) and seed the language detector."""
    global _ready
    if _ready:
        return
    import langdetect
    import nltk

    langdetect.DetectorFactory.seed = 0
    instructions_util.download_nltk_resources()
    missing = []
    for path, name in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            f"instruction following needs the NLTK data {missing}, which is not installed and could not "
            "be downloaded (network access is needed once). Install it with: "
            f"python -m nltk.downloader {' '.join(missing)}"
        )
    _ready = True


def loose_variants(response: str) -> list[str]:
    """The eight variants of ``response`` the loose check tries, in the released code's order."""
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    base = [response, without_first, without_last, without_both]
    return [response, response.replace("*", ""), without_first, without_last, without_both,
            *[v.replace("*", "") for v in base[1:]]]


def check_instruction(instruction_id: str, kwargs: dict | None, prompt: str, response: str) -> tuple[bool, bool]:
    """``(strict, loose)`` result of one instruction on ``response``."""
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=prompt)
    strict = bool(response.strip() and instruction.check_following(response))
    loose = strict or any(v.strip() and instruction.check_following(v) for v in loose_variants(response))
    return strict, loose


@scorer(metrics=[instruction_accuracy()])
def instruction_following() -> Scorer:
    """Score a response against the instructions in the sample's metadata, strict and loose.

    See the module docstring for the checks, the deviations and the NLTK data this needs.
    """

    async def score(state: TaskState, target: Target) -> Score:
        _prepare()
        ids = list(state.metadata["instruction_id_list"])
        kwargs = state.metadata.get("kwargs") or [{}] * len(ids)
        unknown = [i for i in ids if i not in instructions_registry.INSTRUCTION_DICT]
        if unknown:
            raise ValueError(f"sample {state.sample_id}: no verifier for instruction ids {unknown}")
        response = state.output.completion or ""
        truncated = is_truncated(state.output)
        strict, loose, errors = [], [], []
        for instruction_id, kw in zip(ids, kwargs):
            if truncated:
                s = l = False
            else:
                try:
                    s, l = check_instruction(instruction_id, kw, state.input_text, response)
                except Exception as e:  # a verifier bug on odd text must not end the run
                    logger.warning("verifier %s raised %r; counted as not followed", instruction_id, e)
                    errors.append(f"{instruction_id}: {e!r}")
                    s = l = False
            strict.append(s)
            loose.append(l)
        md = {STRICT: strict, LOOSE: loose, "instruction_ids": ids, TRUNCATED: truncated,
              EXTRACTED: response if response.strip() else None, "epoch": state.epoch}
        if errors:
            md["errors"] = errors
        followed = ", ".join(f"{i}: {'pass' if s else 'FAIL'}/{'pass' if l else 'FAIL'}"
                             for i, s, l in zip(ids, strict, loose))
        return Score(value=1.0 if all(strict) else 0.0, answer=response,
                     explanation=f"strict/loose per instruction: {followed}", metadata=md)

    return score
