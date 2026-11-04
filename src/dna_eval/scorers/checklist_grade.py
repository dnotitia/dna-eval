"""``checklist_grade``: judge an answer against a per-question checklist, as HAERAE-VISION does.

The ``grader`` model reads the question, the answer and the checklist, decides for each item
whether the answer meets it (0, 0.5 or 1), and ends with ``<score>K/N</score>``; the score is
K/N. The judge prompt, the way it is assembled and the way the score is read are those of
HAE-RAE/HAERAE-VISION (``utils/evaluate.py``, ``score.py``).
"""
from __future__ import annotations

import asyncio
import re
from fractions import Fraction

from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.judge_failures import JUDGE_ERROR, judge_failures
from dna_eval.utils.grader import call_grader
from dna_eval.utils.model_output import is_truncated

# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2026 HAERAE Team
# From github.com/HAE-RAE/HAERAE-VISION (utils/evaluate.py); see THIRD_PARTY_NOTICES.md.
# The template is sent as published: its placeholders ({{QUESTION}} ...) are not filled in;
# the question, answer and checklist follow it (``build_prompt``), as in score.py.
prompt_template = """[GOAL]
Given a **Question**, **Response**, and a natural-language **Checklist**, decide for each checklist item whether the Response **explicitly** satisfies it: **met = 1**, **not met = 0**. Final score = **(# met) / (total checklist items)**.

[INPUT]
[Question]
{{QUESTION}}

[Response]
{{RESPONSE}}

[Checklist]
{{CHECKLIST}}  ← JSON array or a plain list string. Treat each string as one criterion. Strip any leading numbering like “1.” or “2)”.

[DECISION RULES]
1. **Use only the Response text.** No outside knowledge/assumptions. If uncertain → 0.
2. **Explicitness (mentions/explains/indicates).**
   * 1: Clear, direct statement that fulfills the criterion.
   * 0.5: Indirect/implicit mention that likely implies fulfillment, but not explicit.
   * 0: Not mentioned or contradicted.
   
3. **“All / every / complete” requirements.**
   * 1: Explicitly states completeness (e.g., “all”, “every”, or equivalent).
   * 0.5: Strongly suggests near-completeness (“fill the nests” without “all”, “almost all”).
   * 0: No completeness requirement or states partial suffices.
   
4. **Method / Procedure (“explains how / method”).**
   * 1: Concrete, actionable steps or clear guidance.
   * 0.5: Vague or partial steps (general approach without specifics).
   * 0: No method/procedure provided.
   
5. **“Various / multiple types.”**
   * 1: Names **≥2 distinct, specific types**.
   * 0.5: Mentions variety without naming types, or names only **1** type.
   * 0: No indication of multiple types.
   
6. **Synonyms.** Accept unambiguous equivalents (e.g., “baby dragon” = “hatchling”).
   * 1: Unambiguous equivalence.
   * 0.5: Likely equivalent but slightly ambiguous.
   * 0: Ambiguous or different meaning.

[Evidence policy]
* For **met = 1 or 0.5**, include a **10–60 character direct quote** from the Response supporting the decision.
* For **met = 0 ** include a brief explanation why the response fails the given criteria.
* In the evidence block, list **evidence first**, then the explanation, then the met value.

[OUTPUT FORMAT — STRICT. NO PROSE OUTSIDE TAGS.] <evidence>
Item 1:
evidence: "…direct quote from Response…"
explanation: Briefly justify why criterion 1 earned 1/0.5/0 (reference rule numbers if helpful).
met: 0 | 0.5 | 1

Item 2:
evidence: "…"
explanation: …
met: 0 | 0.5 | 1

… (repeat for all checklist items, in order) </evidence>
<score>
K/N 
</score>

[NOTES]
* Output **only** the two tags above; no code fences, no extra text."""

# Matches either a fraction (e.g., 0.5/5) or a single number (e.g., 0.25) inside <score> tags.
SCORE_RE = re.compile(
    r"""<score>\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:/\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*)?</score>""",
    re.IGNORECASE | re.DOTALL,
)
# SPDX-SnippetEnd

SCORE = "checklist_score"      # Score.metadata key: the K/N value read from the reply


def build_prompt(question: str, answer: str, checklist: list[str]) -> str:
    """The judge prompt as score.py assembles it. The checklist is written as a Python list; the
    official script passes the CSV text of the same list."""
    return f"{prompt_template}[INPUT]\n{question}\n[Response]\n{answer}\n[Checklist]\n{checklist}\n"


def read_score(reply: str) -> float | None:
    """The value of the single ``<score>`` tag in ``reply`` (``K/N`` or a number), as
    ``parse_single_score`` reads it; ``None`` when there is no tag, more than one, or a zero
    denominator."""
    matches = SCORE_RE.findall(reply)
    if len(matches) != 1:
        return None
    num, den = matches[0]
    num_f = Fraction(num).limit_denominator(100000)
    if not den:
        return float(num_f)
    den_f = Fraction(den).limit_denominator(100000)
    return None if den_f == 0 else float(num_f / den_f)


@scorer(metrics=[mean(), stderr(), answer_diagnostics(), judge_failures()])
def checklist_grade(question_key: str = "question", checklist_key: str = "checklist",
                    judge_retries: int = 2) -> Scorer:
    """Score the K/N value the grader gives the final content (never the reasoning) against
    ``state.metadata[checklist_key]``; the question shown is ``state.metadata[question_key]``.

    - A truncated or empty answer scores 0 without asking the grader.
    - The grader is called with the server's default sampling (score.py passes no
      temperature). A failed call, or a reply without exactly one readable ``<score>`` tag, is
      retried ``judge_retries`` times; still failing, it scores 0 and is recorded as
      ``JUDGE_ERROR`` (``"call"`` or ``"parse"``) and counted by ``judge_failures``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion.strip()
        truncated = is_truncated(state.output)
        md: dict = {TRUNCATED: truncated, EXTRACTED: answer or None}
        if truncated or not answer:
            return Score(value=0.0, answer=answer, metadata=md,
                         explanation="truncated" if truncated else "empty answer")
        prompt = build_prompt(state.metadata[question_key], answer, list(state.metadata[checklist_key]))
        reply, value, error = "", None, ""
        for attempt in range(1 + judge_retries):       # a failed call or an unreadable reply: judge again
            reply, error = await call_grader([ChatMessageUser(content=prompt)], GenerateConfig(), 0)
            value = None if error else read_score(reply)
            if value is not None:
                break
            error = error or "parse"
            if attempt < judge_retries:
                await asyncio.sleep(2 ** attempt)
        if value is None:
            md[JUDGE_ERROR] = error
        else:
            md[SCORE] = value
        return Score(value=value or 0.0, answer=answer, metadata=md,
                     explanation=(f"judge {md[JUDGE_ERROR]} failure; " if value is None else "") + reply[-400:])

    return score
