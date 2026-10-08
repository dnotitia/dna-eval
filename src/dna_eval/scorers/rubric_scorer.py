"""``rubric_scorer``: one judge pass against one rubric."""
from __future__ import annotations

from inspect_ai.model import ContentReasoning, GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from pydantic import BaseModel

from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.prompts import build_prompt
from dna_eval.rubrics import Rubric, load_rubric
from dna_eval.utils.json import extract_json


class _Verdict(BaseModel):
    """DeepEval's ``ReasonScore``: both fields required, ``score`` coerced to float."""
    reason: str
    score: float


def read_verdict(reply: str, rubric: Rubric) -> tuple[float, float, str]:
    """(normalised score, raw score, reason) from a judge reply, read as DeepEval's GEval reads it.

    Raises ``ValueError`` (``pydantic.ValidationError`` included) when the reply cannot be
    read. The score is not clipped to the rubric range, as in DeepEval.
    """
    v = _Verdict.model_validate(extract_json(reply))
    return (v.score - rubric.lo) / (rubric.hi - rubric.lo), v.score, v.reason


def content_as_sent(output: ModelOutput) -> str:
    """The judge message's ``content`` as the server sent it, which is what DeepEval reads.

    Inspect moves a ``<think>...</think>`` block out of the text into a ``ContentReasoning``
    (``internal="think"``). DeepEval parses the raw content, so a brace inside that block
    makes it fail; put the block back so ``read_verdict`` fails the same way. Reasoning sent
    in a separate field (``reasoning_content``) is left out, as DeepEval never sees it.
    """
    text = output.completion
    if not output.choices or isinstance(output.message.content, str) or text.lstrip().startswith("<think"):
        return text
    think = "".join(f"<think>{c.reasoning}</think>" for c in output.message.content
                    if isinstance(c, ContentReasoning) and c.internal == "think")
    return think + text


@scorer(metrics=[mean(), per_run_trimmed()])
def rubric_scorer(rubric: str, judge_retries: int = 2) -> Scorer:
    """Judge ``state.output`` against ``rubrics/<rubric>.yaml`` with the model bound to the ``grader`` role.

    The judge is called at temperature 0 and is expected to answer ``{"score": lo-hi, "reason": ...}``.
    The score is normalised to 0-1; ``metadata["passed"]`` applies the rubric threshold;
    ``metadata["epoch"]`` lets ``per_run_trimmed`` regroup scores by run. A judge call that
    fails or cannot be read is retried ``judge_retries`` times and then scored 0, so the
    sample still counts in every average.
    """
    r = load_rubric(rubric)

    async def score(state: TaskState, target: Target) -> Score:
        prompt = build_prompt(
            r, input_text=state.input_text, actual_output=state.output.completion,
            expected_output=target.text, context=state.metadata.get("context"),
        )
        grader = get_model(role="grader", config=GenerateConfig(temperature=0.0))
        last, error = "", ""
        for _ in range(1 + judge_retries):
            try:
                last = content_as_sent(await grader.generate(prompt))
                norm, raw, reason = read_verdict(last, r)
                break
            except ValueError:
                error = "parse"
            except Exception as e:  # noqa: BLE001  (any failed call is retried)
                last, error = f"{type(e).__name__}: {e}", "call"
        else:
            return Score(value=0.0, answer=None, explanation=f"judge failed: {last[:300]}",
                         metadata={"passed": False, "judge_error": error, "epoch": state.epoch})
        return Score(value=norm, answer=f"{raw:g}", explanation=reason,
                     metadata={"passed": norm >= r.threshold, "raw_score": raw, "epoch": state.epoch})

    return score
