"""Every shipped rubric is an Inspect scorer under its own name, re-usable from the CLI."""
import itertools
import subprocess
import sys
from importlib.metadata import entry_points

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import registry_lookup
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate

from dna_eval.rubrics import rubric_names
from dna_eval.scorers.rubric_scorer import rubric_scorer

INSTALLED = any(ep.name == "dna_eval" for ep in entry_points(group="inspect_ai"))


@pytest.mark.parametrize("name", rubric_names())
def test_rubric_is_registered_under_its_name(name):
    # Inspect prefixes the package name only for an installed package; when the tests import
    # the sources directly (pytest's pythonpath) the same scorer is registered without it.
    assert (registry_lookup("scorer", f"dna_eval/{name}") or registry_lookup("scorer", name)) is not None


def test_scores_are_reported_under_rubric_names(tmp_path):
    log = _log(tmp_path)
    assert [s.name for s in log.eval.scorers] == ["language_confusion", "repetition"]
    assert set(log.samples[0].scores) == {"language_confusion", "repetition"}


@pytest.mark.skipif(not INSTALLED, reason="needs the package installed (pip install -e .) for its entry point")
def test_log_can_be_rescored_in_a_fresh_process(tmp_path):
    """`inspect score <log> --scorer dna_eval/<rubric>` finds the scorer without the task."""
    src = _log(tmp_path).location
    out = tmp_path / "rescored.eval"
    proc = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "score", src, "--scorer", "dna_eval/repetition",
         "--action", "overwrite", "--model-role", "grader=mockllm/model",
         "--output-file", str(out), "--display", "none"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rescored = read_eval_log(str(out))
    assert all("repetition" in s.scores for s in rescored.samples)


def _log(tmp_path):
    mk = lambda text: get_model("mockllm/model", custom_outputs=(  # noqa: E731
        ModelOutput.from_content("mockllm/model", text) for _ in itertools.count()))
    task = Task(dataset=[Sample(input="물의 화학식은?")], solver=generate(),
                scorer=[rubric_scorer("language_confusion"), rubric_scorer("repetition")])
    return inspect_eval(task, model=mk("H₂O입니다."), model_roles={"grader": mk('{"score": 9, "reason": "ok"}')},
                        display="none", log_dir=str(tmp_path / "logs"))[0]


def test_store_source_reaches_the_judge(tmp_path):
    """``reasoning_style`` reads ``store.reasoning``: what dna_generate stored, placeholder included."""
    from dna_eval.solvers.dna_generate import EMPTY_REASONING, dna_generate

    mk = lambda text: get_model("mockllm/model", custom_outputs=(  # noqa: E731
        ModelOutput.from_content("mockllm/model", text) for _ in itertools.count()))
    task = Task(dataset=[Sample(input="q", metadata={"reference_reasoning": "참조 생각"})],
                solver=dna_generate(), scorer=rubric_scorer("reasoning_style"))
    log = inspect_eval(task, model=mk("생각 없이 답만"), model_roles={"grader": mk('{"score": 5, "reason": "ok"}')},
                       display="none", log_dir=str(tmp_path / "logs"))[0]
    sample = log.samples[0]
    judge = next(e for e in sample.events if e.event == "model" and e.role == "grader")
    prompt = sample.attachments.get(judge.input[0].text.removeprefix("attachment://"), judge.input[0].text)
    assert f"Actual Output:\n{EMPTY_REASONING} \n" in prompt and "Expected Output:\n참조 생각 \n" in prompt
