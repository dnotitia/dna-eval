"""The rubric judge: the prompt it reads, the call, and how its verdict is read.

``build_prompt`` renders the DeepEval GEval results prompt for a rubric, byte for byte, so scores
stay comparable with DeepEval GEval; the prompt is fixed for the whole suite, and changing it is
a scorer version bump, not a rubric edit. This module makes the call, reads the reply the way
DeepEval's GEval reads it, and runs the re-judge rounds. ``scorers/rubric_scorer.py`` turns the
judgments into Inspect scores.
"""
from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from pydantic import BaseModel

from dna_eval.roles import GRADER_ROLE
from dna_eval.rubrics import PARAM_LABEL, Rubric
from dna_eval.utils.json import escape_stray_backslashes, extract_json
from dna_eval.utils.model_output import content_as_sent, reasoning_text
from dna_eval.utils.templates import render


# DeepEval GEvalTemplate.generate_evaluation_results, reproduced byte for byte (Jinja).
# Source: confident-ai/deepeval (Apache-2.0, Copyright 2024 Confident AI Inc.); see NOTICE.
# Indentation and blank lines are what textwrap.dedent leaves behind in DeepEval; keep them.
# Changing this text changes every score in the suite.
GEVAL_RESULTS = r"""You are an evaluator. Given the following evaluation steps and rubric, assess the response below and return a JSON object with two fields:

            - `"score"`: an integer between {{ rubric.lo }} and {{ rubric.hi }}, based on the rubric provided.
            - `"reason"`: a brief explanation for why the score was given. This must mention specific strengths or shortcomings, referencing relevant details from the input. Do **not** quote the score itself in the explanation.

            Your explanation should:
            - Be specific and grounded in the evaluation steps and rubric.
            - Mention key details from the test case parameters.
            - Be concise, clear, and focused on the evaluation logic.


            Only return valid JSON. Do **not** include any extra commentary or text.

            ---

            Evaluation Steps:
            {% for step in rubric.steps %}{{ loop.index }}. {{ step }}
{% endfor %}

            Rubric:
{% for band in rubric.rubric %}{{ band.range[0] }}{% if band.range[0] != band.range[1] %}-{{ band.range[1] }}{% endif %}: {{ band.outcome }}{% if not loop.last %}
{% endif %}{% endfor %}

            Test Case:
            {% for p in rubric.params %}{{ labels[p] }}:
{{ values[p] }} 

{% endfor %}

            Parameters:
            {{ parameters }}


            ---
            **Example JSON:**
            {% raw %}{{% endraw %}
                "reason": "your concise and informative reason here",
                "score": {{ rubric.lo }}
            {% raw %}}{% endraw %}

            JSON:
"""


def join_english(items) -> str:
    """DeepEval construct_g_eval_params_string: "A", "A and B", "A, B, and C"."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_prompt(rubric: Rubric, *, input_text: str, actual_output: str,
                 expected_output: str = "", context: list[str] | None = None) -> str:
    """The judge prompt for ``rubric``. ``context`` is rendered with ``repr`` as DeepEval does."""
    values = {
        "input": input_text,
        "actual_output": actual_output,
        "expected_output": expected_output,
        "context": repr(context or []),
    }
    return render(GEVAL_RESULTS, rubric=rubric, labels=PARAM_LABEL, values=values,
                  parameters=join_english(PARAM_LABEL[p] for p in rubric.params))



class _Verdict(BaseModel):
    """DeepEval's ``ReasonScore``: both fields required, ``score`` coerced to float."""
    reason: str
    score: float


def read_verdict(reply: str, rubric: Rubric) -> tuple[float, float, str]:
    """(normalised score, raw score, reason) from a judge reply, read as DeepEval's GEval reads it,
    with one exception (see ``docs/methodology.md``): a reply DeepEval cannot parse is read
    again with its stray backslashes escaped, so a verdict quoting LaTeX (``\\pi``) is not lost.

    Raises ``ValueError`` (``pydantic.ValidationError`` included) when the reply cannot be
    read. The score is not clipped to the rubric range, as in DeepEval.
    """
    try:
        data = extract_json(reply)
    except ValueError:
        data = extract_json(escape_stray_backslashes(reply))
    v = _Verdict.model_validate(data)
    return (v.score - rubric.lo) / (rubric.hi - rubric.lo), v.score, v.reason


@dataclass
class Judgment:
    """One judge pass: the normalised score, or the reason it failed."""
    rubric: Rubric
    value: float = 0.0
    raw: float | None = None
    reason: str = ""
    error: str = ""          # "", "parse" or "call"

    @property
    def passed(self) -> bool:
        return not self.error and self.value >= self.rubric.threshold


def judge_field(r: Rubric, param: str, state: TaskState, target: Target, context: list[str] | None):
    """The value of judge-visible field ``param``, from where the rubric's ``sources`` say."""
    src = r.source(param)
    if src == "input":
        return state.input_text
    if src == "answer":
        return state.output.completion
    if src == "reasoning":
        return reasoning_text(state.output)
    if src == "target":
        return target.text
    if src.startswith("store."):
        return state.store.get(src.removeprefix("store."))
    value = state.metadata.get(src.removeprefix("metadata."))
    return context if value is None and param == "context" else value


async def judge_once(r: Rubric, state: TaskState, target: Target, context: list[str] | None) -> Judgment:
    """One judge call for one rubric, with the model bound to the ``grader`` role (required).
    A failed or unreadable call is returned as a failed ``Judgment``, not raised.

    The verdict is read from the reply content as the server sent it (``content_as_sent``):
    DeepEval parses the raw content, so a ``<think>`` block Inspect split off must count.
    """
    f = lambda p: judge_field(r, p, state, target, context)  # noqa: E731
    prompt = build_prompt(r, input_text=f("input"), actual_output=f("actual_output"),
                          expected_output=f("expected_output") or "", context=f("context"))
    grader = get_model(role=GRADER_ROLE, required=True, config=GenerateConfig(temperature=0.0))
    last = ""
    try:
        last = content_as_sent(await grader.generate(prompt))
        value, raw, reason = read_verdict(last, r)
        return Judgment(r, value, raw, reason)
    except ValueError:
        return Judgment(r, reason=f"judge failed: {last[:300]}", error="parse")
    except Exception as e:  # noqa: BLE001  (any failed call is retried)
        return Judgment(r, reason=f"judge failed: {type(e).__name__}: {e}"[:300], error="call")


async def judge_rounds(parts: list[Rubric], state: TaskState, target: Target,
                       judge_retries: int, context: list[str] | None) -> list[Judgment]:
    """Judge every rubric; if any judgment failed, judge them all again, up to ``judge_retries``
    more rounds (as DeepEval re-evaluates a whole failed test case). Judgments still failing
    after the last round score 0, so the sample still counts in every average."""
    for _ in range(1 + judge_retries):
        judgments = [await judge_once(r, state, target, context) for r in parts]
        if not any(j.error for j in judgments):
            break
    return judgments
