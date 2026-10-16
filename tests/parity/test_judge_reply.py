"""A judge reply text is read exactly as DeepEval's GEval reads it (deepeval 4.0.3), except
for the replies in ``DEVIATIONS``.

``fixtures/judge_outputs.json`` holds raw judge replies and what GEval made of them: the
normalised score and reason, or an error. It is produced by running the replies through
DeepEval itself (see ``fixtures/make_judge_outputs.py``).
"""
import json
from pathlib import Path

import pytest

from dna_eval.rubrics import load_rubric
from dna_eval.rubric_judging import read_verdict

FIXTURE = Path(__file__).parent.parent / "fixtures" / "judge_outputs.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


# Replies DeepEval rejects but this project reads, on purpose (docs/methodology.md, "판정"): a single
# backslash that is not a JSON escape, as in LaTeX quoted inside the reason.
DEVIATIONS = {
    "latex_invalid_escape": (1.0, "수식($n, \\mid$, \\pi)만 사용됨"),
    "latex_bad_unicode_escape": (0.7, "\\underline{x} 사용"),
}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_judge_reply_matches_deepeval(case):
    rubric = load_rubric("performance")   # 0-10, the score range the fixture was made with
    expected = case["expected"]
    if case["id"] in DEVIATIONS:
        assert "error" in expected                       # DeepEval rejects it ...
        assert read_verdict(case["raw"], rubric)[::2] == DEVIATIONS[case["id"]]   # ... we read it
        return
    if "error" in expected:
        with pytest.raises(ValueError):
            read_verdict(case["raw"], rubric)
    else:
        score, _, reason = read_verdict(case["raw"], rubric)
        assert (score, reason) == (expected["score"], expected["reason"])
