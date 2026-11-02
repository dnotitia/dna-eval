"""Read and compare final answers to math problems written in LaTeX.

Pieces, from the most to the least literal:

- ``last_boxed``: the content of the last ``\\boxed{...}`` or ``\\fbox{...}`` in a response,
  with nested braces handled by counting (the approach MathArena's parser takes with a
  recursive pattern).
- ``last_integer``, ``parse_integer``: the last whole number in a text, and a strict reading
  of an answer string as an integer (``"042"`` and ``"42.0"`` both read as 42).
- ``extract_boxed_answer``: the answer string a response gives, following the lm-evaluation-
  harness HRM8K task: the last box (the text after the last ``=`` inside it), or, with no box
  in the response, the last ``$...$`` span, or the last number.
- ``is_equiv_string``: string equivalence after the normalisation of Hendrycks et al.'s MATH
  evaluation (strip spaces, ``\\left``, units, ``\\dfrac`` -> ``\\frac``, ``0.5`` -> ``\\frac{1}{2}``, ...).
- ``is_equiv_math_verify``: symbolic equivalence through the ``math-verify`` package, which
  understands fractions, radicals, sets, intervals and the like.

The Hendrycks normalisation functions below are ported from lm-evaluation-harness
(``lm_eval/tasks/hrm8k/default/utils.py``, MIT licence, Copyright (c) 2020 EleutherAI), which
in turn took them from https://github.com/hendrycks/math (``modeling/math_equivalence.py``,
MIT licence). Their bodies are kept as published, comments included, so results match.
"""
from __future__ import annotations

import re
import threading

# --- boxed answers ------------------------------------------------------------------------

_BOX_START = re.compile(r"\\(?:boxed|fbox)\s*\{")


def _braced(text: str, open_idx: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``open_idx``, or ``None`` if it is never closed."""
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _unbox(content: str) -> str:
    """``content`` with any ``\\boxed{X}`` / ``\\fbox{X}`` inside it replaced by ``X``."""
    while True:
        m = _BOX_START.search(content)
        if m is None:
            return content
        end = _braced(content, m.end() - 1)
        if end is None:
            return content
        content = content[:m.start()] + content[m.end():end] + content[end + 1:]


def last_boxed(text: str) -> str | None:
    """Content of the last complete ``\\boxed{...}`` or ``\\fbox{...}`` in ``text``, or ``None``.

    Braces may nest (``\\boxed{\\frac{1}{2}}``); a box inside a box is unwrapped. A box that is
    never closed (a cut-off response) is ignored in favour of an earlier complete one.
    """
    last = None
    pos = 0
    while (m := _BOX_START.search(text, pos)) is not None:
        end = _braced(text, m.end() - 1)
        if end is None:
            pos = m.end()
            continue
        last = text[m.end():end]
        pos = end + 1
    return None if last is None else _unbox(last).strip()


# --- integers -----------------------------------------------------------------------------

def last_integer(text: str) -> int | None:
    """The last whole number in ``text`` (digits delimited by word boundaries), or ``None``."""
    matches = re.findall(r"\b\d+\b", text)
    return int(matches[-1]) if matches else None


def parse_integer(answer: str) -> int | None:
    """Read ``answer`` as an integer, or ``None`` if it is not one.

    Surrounding ``$``, spaces, ``\\text{}`` wrappers and thousands commas are ignored, and a
    decimal that is a whole number counts (``"42.0"``), so ``"042"`` and ``"42.0"`` give 42.
    """
    s = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", answer)
    s = s.replace("\\!", "").replace("\\,", "").replace("$", "").replace(" ", "").replace(",", "")
    return int(float(s)) if re.fullmatch(r"[+-]?\d+(\.0*)?", s) else None


# --- answer string, as the lm-evaluation-harness HRM8K task reads it ------------------------

def extract_boxed_answer(response: str) -> str | None:
    """The answer string of ``response``, as the lm-evaluation-harness HRM8K task takes it.

    - With a ``\\boxed``/``\\fbox`` in the response: the content of the last complete one, cut
      to the text after its last ``=`` (``\\boxed{x = 5}`` gives ``5``); ``None`` if the box
      is never closed, with no fallback.
    - Without one: the last ``$...$`` span of a line (again after its last ``=``); else the
      text after the last ``=`` of the response, else its last number.
    """
    if "\\boxed" in response or "\\fbox" in response:
        box = last_boxed(response)
        if box is None:
            return None
        return box.split("=")[-1].lstrip(" ") if "=" in box else box
    dollars = re.findall(r"\$(.*)\$", response)
    if dollars:
        span = dollars[-1]
        return span.split("=")[-1].lstrip(" ") if "=" in span else span
    if "=" in response:
        tail = response.split("=")[-1].lstrip(" ").rstrip(".")
        return tail.split("\\n")[0] if "\\n" in tail else tail
    numbers = re.findall(r"(?:\$)?\d+(?:\.\d+)?(?![\w\d])", response)
    return numbers[-1] if numbers else None


def normalise_gold(answer: object) -> str:
    """The reference answer as a string: whole-number floats lose their ``.0`` (``"42.0"`` -> ``"42"``)."""
    s = str(answer).strip()
    try:
        value = float(s)
        return str(int(value)) if value.is_integer() else str(value)
    except Exception:
        return s


def is_equiv_string(answer: str | None, gold: object) -> bool:
    """Whether ``answer`` equals ``gold`` after the Hendrycks et al. string normalisation.

    The reference is normalised directly rather than read as a response, which would reduce a
    non-numeric reference such as ``8\\pi`` to its last number.
    """
    if answer is None:
        return False
    gold = normalise_gold(gold)
    try:
        a = normalise_gold(_strip_string(answer))
        return a == _strip_string(gold)
    except Exception:
        return answer == gold


# --- symbolic equivalence ------------------------------------------------------------------

def is_equiv_math_verify(answer: str | None, gold: object) -> bool:
    """Whether ``math-verify`` finds ``answer`` mathematically equal to ``gold`` (``False`` on any failure).

    Both are read as LaTeX (``$...$``); ``math-verify`` also accepts plain expressions such
    as ``1/6``. Its parse and compare steps time out after 5 seconds, which needs the main
    thread, so off the main thread they run without a timeout.
    """
    if answer is None or not str(answer).strip():
        return False
    from math_verify import parse, verify   # imported here: the package is only needed by the math tasks

    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    try:
        g = parse(f"${normalise_gold(gold)}$", parsing_timeout=timeout)
        a = parse(f"${answer}$", parsing_timeout=timeout)
        return bool(g) and bool(a) and verify(g, a, timeout_seconds=timeout)   # verify(gold, answer)
    except Exception:
        return False


# --- Hendrycks et al. normalisation (ported; see the module docstring) ------------------------

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
