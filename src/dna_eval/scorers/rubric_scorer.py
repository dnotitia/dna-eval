"""``rubric_scorer``: one judge pass against one rubric."""
from __future__ import annotations

import json

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from dna_eval.metrics.per_run_trimmed import per_run_trimmed
from dna_eval.prompts import build_prompt
from dna_eval.rubrics import load_rubric
from dna_eval.utils.json import extract_json


@scorer(metrics=[mean(), per_run_trimmed()])
def rubric_scorer(rubric: str, judge_retries: int = 2) -> Scorer:
    """Judge ``state.output`` against ``rubrics/<rubric>.yaml`` with the model bound to the ``grader`` role.

    The judge is called at temperature 0 and is expected to answer ``{"score": lo-hi, "reason": ...}``.
    The score is normalised to 0-1; ``metadata["passed"]`` applies the rubric threshold;
    ``metadata["epoch"]`` lets ``per_run_trimmed`` regroup scores by run. Unparsable judge
    output is retried ``judge_retries`` times and then scored 0, as DeepEval does.
    """
    r = load_rubric(rubric)

    async def score(state: TaskState, target: Target) -> Score:
        prompt = build_prompt(
            r, input_text=state.input_text, actual_output=state.output.completion,
            expected_output=target.text, context=state.metadata.get("context"),
        )
        grader = get_model(role="grader", config=GenerateConfig(temperature=0.0))
        last = ""
        for _ in range(1 + judge_retries):
            last = (await grader.generate(prompt)).completion
            try:
                data = extract_json(last)
                raw, reason = int(data["score"]), str(data.get("reason", ""))
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        else:
            return Score(value=0.0, answer=None, explanation=f"judge parse failed: {last[:300]}",
                         metadata={"passed": False, "judge_error": "parse", "epoch": state.epoch})
        norm = (raw - r.lo) / (r.hi - r.lo)
        return Score(value=norm, answer=str(raw), explanation=reason,
                     metadata={"passed": norm >= r.threshold, "raw_score": raw, "epoch": state.epoch})

    return score
