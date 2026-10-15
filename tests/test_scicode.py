"""SciCode: prompts, code extraction and test scripts against SciCode's own code; the scorer on the
local sandbox with small synthetic targets; the metrics.

``fixtures/scicode/official_prompts.json`` was produced by running the benchmark's generation and
evaluation code (``fixtures/make_scicode_prompts.py``). The scorer tests run scripts with this
interpreter and need ``h5py``, ``scipy`` and ``sympy``; they skip without them.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.solver import TaskState, solver

from dna_eval.metrics.answer_diagnostics import answer_diagnostics
from dna_eval.metrics.sub_step_accuracy import sub_step_accuracy
from dna_eval.scorers import scicode_tests as scorer_module
from dna_eval.scorers.scicode_tests import assemble_script, extract_targets, scicode_tests
from dna_eval.solvers.scicode_steps import (WITH_BACKGROUND_TEMPLATE, WITHOUT_BACKGROUND_TEMPLATE, build_prompt,
                                            extract_function_name, extract_python_script, get_function_from_code,
                                            scicode_steps)
from dna_eval.tasks.benchmarks.scicode import SKIPPED, record_to_sample

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "scicode" / "official_prompts.json").read_text(encoding="utf-8"))
PROBLEMS = {p["problem_id"]: p for p in FIXTURE["problems"]}
H5_PLACEHOLDER = FIXTURE["source"]["h5_placeholder"]
RUNS = [pytest.param(r, id=f"{r['problem_id']}-{'bg' if r['with_background'] else 'nobg'}") for r in FIXTURE["runs"]]


def sample_for(problem_id: str) -> Sample:
    record = {**PROBLEMS[problem_id], "problem_name": "name", "problem_description_main": "main description"}
    return record_to_sample(record)


def run_solver(sample: Sample, replies: dict[str, str], with_background: bool):
    """Run ``scicode_steps`` on ``sample`` with a model that answers each sub-step from ``replies``;
    return the prompts it was sent and the store entries."""
    prompts: list[str] = []

    async def generate(state: TaskState, **kwargs) -> TaskState:
        prompts.append(state.messages[-1].text)
        step = [s for s in sample.metadata["sub_steps"] if s["step_number"] in replies][len(prompts) - 1]
        reply = replies[step["step_number"]]
        state.output = ModelOutput.from_content("mockllm/model", reply)
        state.messages.append(ChatMessageAssistant(content=reply))
        return state

    state = TaskState(model="mockllm/model", sample_id=sample.id, epoch=1, input=sample.input, messages=[],
                      metadata=sample.metadata)
    state = asyncio.run(scicode_steps(with_background)(state, generate))
    return prompts, state.store.get("scicode_steps"), state


# ---------------------------------------------------------------- prompts, extraction, scripts

@pytest.mark.parametrize("run", RUNS)
def test_prompts_code_and_scripts_match_the_official_code(run):
    sample = sample_for(run["problem_id"])
    prompts, entries, _ = run_solver(sample, run["replies"], run["with_background"])
    evaluated = [e for e in entries if not e["skipped"]]
    assert [e["step_number"] for e in evaluated] == list(run["prompts"])
    assert prompts == list(run["prompts"].values())
    for e, step in zip(entries, sample.metadata["sub_steps"]):
        if e["skipped"]:
            continue
        sid = e["step_number"]
        assert f"{e['previous_code']}\n{e['code']}" == run["code_files"][sid]
        script = assemble_script(f"{e['previous_code']}\n{e['code']}", sid, step["test_cases"], H5_PLACEHOLDER)
        assert script == run["test_scripts"][sid]


def test_skipped_steps_are_not_generated_and_their_gold_code_is_context():
    sample = sample_for("76")
    run = next(r for r in FIXTURE["runs"] if r["problem_id"] == "76" and not r["with_background"])
    prompts, entries, state = run_solver(sample, run["replies"], False)
    assert [e["skipped"] for e in entries] == [False, False, True, False]
    assert len(prompts) == 3                                   # no request for 76.3
    gold = entries[2]["code"]
    assert gold.startswith("def generate_dna(")                # found by name in the gold file
    assert gold in prompts[2] and gold in entries[3]["previous_code"]
    assert "76.3" in SKIPPED and "62.1" in SKIPPED and "13.6" in SKIPPED


def test_class_headers_reduce_the_gold_code_to_init_as_upstream_does():
    sample = sample_for("62")
    skipped = sample.metadata["sub_steps"][0]
    assert skipped["skipped_code"] and skipped["function_header"].startswith("class EnlargedBlock")
    assert extract_function_name(skipped["function_header"]) == "__init__"
    gold = get_function_from_code(skipped["skipped_code"], "__init__")
    assert gold.startswith("def __init__(self, length, basis_size, operator_dict):")
    assert "class " not in gold
    assert all(s["skipped_code"] is None for s in sample.metadata["sub_steps"][1:])


def test_transcript_and_store_entries():
    sample = sample_for("14")
    run = next(r for r in FIXTURE["runs"] if r["problem_id"] == "14")
    _, entries, state = run_solver(sample, run["replies"], False)
    assert [m.role for m in state.messages] == ["user", "assistant", "user", "assistant"]
    assert entries[0]["fenced"] and entries[0]["response"] == run["replies"]["14.1"]
    assert not any(e["truncated"] for e in entries)


def test_extract_python_script_follows_the_official_rules():
    assert extract_python_script("text\n```python\nimport os\nx = 1\n```\nmore") == "\nx = 1\n"
    assert extract_python_script("```\ndef f():\n    return 1\n```") == "\ndef f():\n    return 1\n"
    assert extract_python_script("a\n```python\nx = 1\n```\nb\n```python\ny = 2\n```") == "\nx = 1\n"   # first block
    assert extract_python_script("x = 1\nfrom numpy import array\ny = 2") == "x = 1\n\ny = 2"          # no block: all of it
    assert extract_python_script("```python\ndef f():\n    import math\n    return 1\n```") == \
        "\ndef f():\n\n    return 1\n"                                                                  # imports go anywhere


def test_templates_use_the_same_placeholders():
    for template in (WITH_BACKGROUND_TEMPLATE, WITHOUT_BACKGROUND_TEMPLATE):
        for placeholder in ("{problem_steps_str}", "{next_step_str}", "{dependencies}"):
            assert template.count(placeholder) == 1
    assert not WITH_BACKGROUND_TEMPLATE.endswith("\n") and WITHOUT_BACKGROUND_TEMPLATE.endswith("```\n")
    assert "scientific background" in WITHOUT_BACKGROUND_TEMPLATE


def test_build_prompt_first_step_has_no_previous_steps():
    steps = [{"step_description_prompt": "Do A.", "step_background": "BG", "function_header": "def a():",
              "return_line": "return 1"}]
    prompt, previous = build_prompt(steps, "import numpy as np", 0, [""], False)
    assert "PREVIOUS STEPS DESCRIPTION:\n\n\nNEXT STEP" in prompt
    assert "Do A.\n\ndef a():\n\nreturn 1" in prompt and "BG" not in prompt
    assert previous == "import numpy as np\n\n"
    assert "Do A.\nBG\n\ndef a():" in build_prompt(steps, "import numpy as np", 0, [""], True)[0]


def test_assemble_script_layout():
    script = assemble_script("CODE", "1.2", ["a = 1\nassert a", "b = 2"], "T.h5")
    assert script == ("CODE\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n\n"
                      "targets = process_hdf5_to_tuple('1.2', 2, 'T.h5')\n"
                      "target = targets[0]\n\na = 1\nassert a\ntarget = targets[1]\n\nb = 2\n")


# ---------------------------------------------------------------- the scorer, on the local sandbox

def test_missing_h5py_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(RuntimeError, match=r'h5py.*pip install "dna-eval\[scicode\]"'):
        scicode_tests("unused.h5")
    with pytest.raises(RuntimeError, match="h5py"):
        extract_targets("unused.h5", ["1.1"])


@pytest.fixture(scope="module")
def targets_file(tmp_path_factory):
    np = pytest.importorskip("numpy")
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    pytest.importorskip("sympy")
    path = tmp_path_factory.mktemp("scicode") / "targets.h5"
    with h5py.File(path, "w") as f:
        f["S.1/test1/var1"] = np.array([4.0, 9.0])                         # one variable: an array
        f["S.2/test1/var1"] = 3.0                                          # two variables: a tuple
        f["S.2/test1/var2"] = np.array([1.0, 2.0])
        group = f.create_group("S.2/test2/var1")                           # a group of values: a dict
        group["a"] = 1.0
        group["b"] = np.array([1, 2])
        f["S.9/test1/var1"] = 0.0                                          # a step no sample asks about
    return path


def test_extract_targets_copies_only_the_requested_groups(targets_file, tmp_path):
    import h5py
    data = extract_targets(targets_file, ["S.2"])
    out = tmp_path / "out.h5"
    out.write_bytes(data)
    with h5py.File(out, "r") as f:
        assert list(f) == ["S.2"] and set(f["S.2"]) == {"test1", "test2"}
        assert f["S.2/test1/var1"][()] == 3.0
    with pytest.raises(KeyError, match="S.0"):
        extract_targets(targets_file, ["S.0"])


STEP1 = "def square_all(x):\n    return x ** 2\n"
STEP2 = "def pair():\n    return (3.0, np.array([1.0, 2.0]))\n\ndef info():\n    return {'a': 1.0, 'b': np.array([1, 2])}\n"
TESTS1 = ["assert np.allclose(square_all(np.array([2.0, 3.0])), target)"]
TESTS2 = ["from scicode.compare.cmp import cmp_tuple_or_list\nassert cmp_tuple_or_list(pair(), target)",
          "from scicode.compare.cmp import are_dicts_close\nassert are_dicts_close(info(), target)"]
DEPS = "import numpy as np\n"


def entry(step_number, code, previous="", **kw):
    return {"step_number": step_number, "skipped": False, "response": "", "code": code,
            "previous_code": DEPS + previous + ("\n" if previous else ""), "truncated": False, "fenced": True, **kw}


def sub_step(number, tests):
    return {"step_number": number, "test_cases": tests, "skipped_code": None}


CASES = {   # sample id: (sub_steps, entries)
    "ok": ([sub_step("S.1", TESTS1), sub_step("S.2", TESTS2)], [entry("S.1", STEP1), entry("S.2", STEP2, STEP1)]),
    "wrong": ([sub_step("S.1", TESTS1), sub_step("S.2", TESTS2)],
              [entry("S.1", STEP1), entry("S.2", STEP2.replace("3.0", "3.5"), STEP1)]),
    "slow": ([sub_step("S.1", TESTS1)], [entry("S.1", "import time\ntime.sleep(60)\n" + STEP1)]),
    "truncated": ([sub_step("S.1", TESTS1)], [entry("S.1", STEP1, truncated=True)]),
    "unfenced": ([sub_step("S.1", TESTS1)], [entry("S.1", STEP1, fenced=False)]),
    "skipped": ([sub_step("S.1", TESTS1), {**sub_step("S.2", []), "skipped_code": "x"}],
                [entry("S.1", STEP1), {**entry("S.2", ""), "skipped": True}]),
    "syntax": ([sub_step("S.1", TESTS1)], [entry("S.1", "def square_all(x:\n")]),
}


@solver
def planted_steps():
    """Stands in for ``scicode_steps``: the entries a case plants in the metadata go to the store."""
    async def solve(state, generate):
        state.store.set("scicode_steps", state.metadata["entries"])
        return state
    return solve


@pytest.fixture(scope="module")
def scored(targets_file, tmp_path_factory):
    samples = [Sample(id=name, input="x", target="", metadata={"sub_steps": steps, "entries": entries})
               for name, (steps, entries) in CASES.items()]
    task = Task(dataset=samples, solver=planted_steps(),
                scorer=scicode_tests(targets_file, timeout=5, python=sys.executable), sandbox="local")
    log = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path_factory.mktemp("logs")))[0]
    assert log.status == "success", log.error
    return log, {s.id: s.scores["scicode_tests"] for s in log.samples}


def test_scorer_results_per_problem(scored):
    _, scores = scored
    value = {k: v.value for k, v in scores.items()}
    assert value == {"ok": 1.0, "wrong": 0.0, "slow": 0.0, "truncated": 0.0, "unfenced": 1.0, "skipped": 1.0,
                     "syntax": 0.0}
    status = {k: [s["status"] for s in v.metadata["steps"]] for k, v in scores.items()}
    assert status == {"ok": ["pass", "pass"], "wrong": ["pass", "fail"], "slow": ["timeout"],
                      "truncated": ["truncated"], "unfenced": ["pass"], "skipped": ["pass"], "syntax": ["fail"]}
    assert "AssertionError" in scores["wrong"].metadata["steps"][1]["stderr"]
    assert "SyntaxError" in scores["syntax"].metadata["steps"][0]["stderr"]
    assert scores["wrong"].answer == "1/2 sub-steps"


def test_scorer_metadata_for_the_metrics(scored):
    _, scores = scored
    md = {k: v.metadata for k, v in scores.items()}
    assert (md["ok"]["steps_passed"], md["ok"]["steps_total"]) == (2, 2)
    assert (md["wrong"]["steps_passed"], md["wrong"]["steps_total"]) == (1, 2)
    assert (md["skipped"]["steps_passed"], md["skipped"]["steps_total"]) == (1, 1)      # the skipped step is not counted
    assert md["truncated"]["truncated"] is True and md["ok"]["truncated"] is False
    assert md["unfenced"]["extracted"] is None and md["ok"]["extracted"] == "2 code blocks"


def test_scorer_metrics_in_the_log(scored):
    log, _ = scored
    metrics = {m: v.value for s in log.results.scores for m, v in s.metrics.items()}
    assert metrics["accuracy"] == pytest.approx(3 / 7)
    assert metrics["sub_step_accuracy"] == pytest.approx(5 / 9)       # 2+1+0+0+1+1+0 passed of 2+2+1+1+1+1+1
    assert metrics["truncated"] == pytest.approx(1 / 7)
    assert metrics["format_failure"] == pytest.approx(1 / 7)


def test_scorer_requires_the_solver(targets_file, tmp_path):
    task = Task(dataset=[Sample(id=1, input="x", metadata={"sub_steps": []})], solver=[],
                scorer=scicode_tests(targets_file, python=sys.executable), sandbox="local")
    log = inspect_eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path), fail_on_error=False)[0]
    assert log.samples[0].error is not None and "scicode_steps" in log.samples[0].error.message


def test_the_scorer_files_in_the_sandbox_are_the_vendored_helpers():
    vendor = scorer_module.VENDOR
    assert (vendor / "cmp.py").is_file() and (vendor / "parse.py").is_file()
    assert "from datasets import" not in "\n".join(
        line for line in (vendor / "parse.py").read_text().splitlines() if not line.startswith(" "))   # imported on use


# ---------------------------------------------------------------- metrics

def _score(passed, total, truncated=False, extracted="x", value=None):
    value = float(passed == total) if value is None else value
    return SampleScore(score=Score(value=value, metadata={"steps_passed": passed, "steps_total": total,
                                                          "truncated": truncated, "extracted": extracted}))


def test_sub_step_accuracy_pools_sub_steps_over_runs():
    compute = sub_step_accuracy()
    # three runs of two problems with 4 and 1 evaluated sub-steps
    scores = [_score(4, 4), _score(1, 1), _score(2, 4), _score(1, 1), _score(0, 4), _score(0, 1)]
    assert compute(scores) == pytest.approx(8 / 15)
    assert compute([]) != compute([])                                 # NaN without scores
    assert compute([_score(3, 4)]) == pytest.approx(0.75)


def test_problem_accuracy_differs_from_sub_step_accuracy():
    scores = [_score(4, 4), _score(2, 4), _score(1, 1)]
    assert sum(s.score.value for s in scores) / 3 == pytest.approx(2 / 3)
    assert sub_step_accuracy()(scores) == pytest.approx(7 / 9)


def test_answer_diagnostics_reads_the_scicode_keys():
    scores = [_score(4, 4), _score(1, 4, truncated=True), _score(0, 4, extracted=None), _score(1, 1)]
    out = answer_diagnostics()(scores)
    assert out["truncated"] == pytest.approx(1 / 4) and out["format_failure"] == pytest.approx(1 / 4)
    assert out["accuracy_excl_truncated"] == pytest.approx(2 / 3)
