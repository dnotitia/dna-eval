"""T2 — the prompt the judge receives is byte-identical to DeepEval's GEval render (deepeval 4.0.3)."""
from pathlib import Path

import pytest

from dna_eval.prompts import build_prompt
from dna_eval.rubrics import load_rubric
from tests.parity.cases import RUBRIC_CASES

FIXTURES = Path(__file__).parent.parent / "fixtures" / "judge_prompts"


@pytest.mark.parametrize("rubric,judge_input", RUBRIC_CASES, ids=[r for r, _ in RUBRIC_CASES])
def test_judge_prompt_matches_deepeval(rubric, judge_input):
    ours = build_prompt(load_rubric(rubric), **judge_input)
    assert ours == (FIXTURES / f"{rubric}.txt").read_text(encoding="utf-8")
