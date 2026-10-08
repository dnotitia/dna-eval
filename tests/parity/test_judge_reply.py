"""A judge reply text is read exactly as DeepEval's GEval reads it (deepeval 4.0.3).

``fixtures/judge_outputs.json`` holds raw judge replies and what GEval made of them: the
normalised score and reason, or an error. It is produced by running the replies through
DeepEval itself (see ``fixtures/make_judge_outputs.py``).
"""
import json
from pathlib import Path

import pytest

from dna_eval.rubrics import load_rubric
from dna_eval.scorers.rubric_scorer import read_verdict

FIXTURE = Path(__file__).parent.parent / "fixtures" / "judge_outputs.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_judge_reply_matches_deepeval(case):
    rubric = load_rubric("performance")   # 0-10, the score range the fixture was made with
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError):
            read_verdict(case["raw"], rubric)
    else:
        score, _, reason = read_verdict(case["raw"], rubric)
        assert (score, reason) == (expected["score"], expected["reason"])
