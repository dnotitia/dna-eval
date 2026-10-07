"""Judge prompts: how a rubric is turned into the text the grader reads.

``geval_results.jinja`` reproduces the DeepEval GEval results prompt byte-for-byte so scores
stay comparable with the DeepEval-based scoreboard. It is fixed for the whole suite; changing
it is a scorer version bump, not a rubric edit.
"""
from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined

from dna_eval.rubrics import PARAM_LABEL, Rubric

GEVAL_RESULTS = "geval_results.jinja"


def join_english(items) -> str:
    """DeepEval construct_g_eval_params_string: "A", "A and B", "A, B, and C"."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


env = Environment(
    loader=PackageLoader("dna_eval", "prompts"),
    undefined=StrictUndefined,       # a typo in the template fails loudly instead of rendering blank
    keep_trailing_newline=True,      # the DeepEval prompt ends with "JSON:\n"
    autoescape=False,
)
env.filters["join_english"] = join_english


def build_prompt(rubric: Rubric, *, input_text: str, actual_output: str,
                 expected_output: str = "", context: list[str] | None = None,
                 template: str = GEVAL_RESULTS) -> str:
    """Render the judge prompt. ``context`` is rendered with ``repr`` as DeepEval does."""
    values = {
        "input": input_text,
        "actual_output": actual_output,
        "expected_output": expected_output,
        "context": repr(context or []),
    }
    return env.get_template(template).render(
        rubric=rubric, labels=PARAM_LABEL, values=values,
        param_labels=[PARAM_LABEL[p] for p in rubric.params],
    )
