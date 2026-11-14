"""run.sh builds the right ``inspect eval`` commands.

A fake ``inspect`` first on PATH records each call's arguments (one JSON list per line), so these
tests check run.sh's argument rules without a model, a judge, an endpoint or the network:

- each group runs its tasks in one call, in order, and style in a second call with
  ``-T reference=<reference-model>``; without a reference model a group skips style and says so;
- style alone needs a reference model; benchmarks, vision and single tasks refuse one;
- uncensorship runs only when named; unknown tasks and too few arguments exit 2 with a message;
- every call names the tasks ``dna_eval/<task>``, passes ``--model`` and ``--model-role grader=<judge>``,
  and forwards the options verbatim (spaces and glob characters included);
- a failing ``inspect`` stops the run with its exit status;
- the task lists below, run.sh and the tasks the package registers name the same tasks.
"""
import ast
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "run.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="run.sh needs bash")

MODEL = "vllm/my-model"
JUDGE = "openai-api/judge/judge-model"
REFERENCE = "Qwen3.5-9B"

# Written out on purpose rather than read from run.sh: a change to run.sh's lists must show up here.
SUITE = ["performance", "hallucination", "persona", "degeneration"]     # style runs on its own
TEXT_BENCHMARKS = ["kmmlu_pro", "kmmlu_redux", "kmmlu_hard", "click", "kobalt", "hrm8k_ksm", "haerae_bench",
                   "kosimpleqa", "kormedmcqa", "kbl"]
VISION = ["haerae_vision", "kmmmu"]
NAMED_ONLY = ["uncensorship"]

# group -> (tasks of the first call, whether the group includes style)
GROUPS = {
    "dna_eval": (SUITE, True),
    "benchmarks": (TEXT_BENCHMARKS + VISION, False),
    "all": (SUITE + TEXT_BENCHMARKS + VISION, True),
    "text_only": (SUITE + TEXT_BENCHMARKS, True),
    "vision": (VISION, False),
}
STYLE_GROUPS = [g for g, (_, style) in GROUPS.items() if style]
NO_STYLE_GROUPS = [g for g, (_, style) in GROUPS.items() if not style]
SINGLE_TASKS = SUITE + TEXT_BENCHMARKS + VISION + NAMED_ONLY      # every task but style

STYLE_SKIPPED = "style skipped: give a reference model to run it"
USAGE = "bash run.sh <task> <model> <judge-model> [reference-model] [inspect eval options...]"

FAKE_INSPECT = """\
import json, os, sys
with open(os.environ["FAKE_INSPECT_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ["FAKE_INSPECT_EXIT"]))
"""


def _registered_tasks() -> set[str]:
    """The task names ``_registry.py`` imports from ``dna_eval.tasks``: what ``dna_eval/<task>`` resolves to."""
    tree = ast.parse((ROOT / "src" / "dna_eval" / "_registry.py").read_text(encoding="utf-8"))
    return {alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("dna_eval.tasks.") for alias in node.names}


REGISTERED = _registered_tasks()


class Result(NamedTuple):
    returncode: int
    calls: list[list[str]]     # one argv (without the program name) per `inspect` call, in order
    stdout: str
    stderr: str


def _call(tasks, *extra, model=MODEL, judge=JUDGE) -> list[str]:
    return ["eval", *(f"dna_eval/{t}" for t in tasks), "--model", model, "--model-role", f"grader={judge}", *extra]


def _group_calls(group, reference=None, options=()) -> list[list[str]]:
    tasks, style = GROUPS[group]
    calls = [_call(tasks, *options)]
    if style and reference:
        calls.append(_call(["style"], "-T", f"reference={reference}", *options))
    return calls


@pytest.fixture(scope="session")
def fake_bin(tmp_path_factory):
    """A directory holding the fake ``inspect``; its path has a space, so run.sh must quote paths."""
    bin_dir = tmp_path_factory.mktemp("fake bin")
    helper = bin_dir / "fake_inspect.py"
    helper.write_text(FAKE_INSPECT, encoding="utf-8")
    fake = bin_dir / "inspect"
    fake.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I -S {shlex.quote(str(helper))} "$@"\n',
                    encoding="utf-8")
    fake.chmod(0o755)
    return bin_dir


@pytest.fixture
def run(tmp_path, fake_bin):
    """``run(*args, inspect_exit=0)`` runs ``bash run.sh *args`` with the fake ``inspect`` and returns a Result."""
    cwd = tmp_path / "work dir"        # run.sh writes nothing here; the space checks quoting again
    cwd.mkdir()
    (cwd / "glob-bait").touch()        # an unquoted `*` in run.sh would expand to this file
    log = tmp_path / "inspect.jsonl"

    def _run(*args, inspect_exit=0) -> Result:
        env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV")}
        env.update(PATH=f"{fake_bin}{os.pathsep}{env.get('PATH', '')}", FAKE_INSPECT_LOG=str(log),
                   FAKE_INSPECT_EXIT=str(inspect_exit))
        proc = subprocess.run([BASH, str(RUN_SH), *args], cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=60)
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        log.unlink(missing_ok=True)
        return Result(proc.returncode, calls, proc.stdout, proc.stderr)

    return _run


def _assert_usage_error(result: Result, message: str):
    assert result.returncode == 2
    assert message in result.stderr
    assert USAGE in result.stdout
    assert result.calls == []


# --- groups ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("group", list(GROUPS))
def test_group_without_reference(run, group):
    result = run(group, MODEL, JUDGE)
    assert result.returncode == 0, result.stderr
    assert result.calls == _group_calls(group)
    assert result.stderr.strip() == (STYLE_SKIPPED if group in STYLE_GROUPS else "")


@pytest.mark.parametrize("group", STYLE_GROUPS)
def test_group_with_reference_runs_style_last(run, group):
    result = run(group, MODEL, JUDGE, REFERENCE)
    assert result.returncode == 0, result.stderr
    assert result.calls == _group_calls(group, REFERENCE)
    assert len(result.calls) == 2 and result.calls[1] == _call(["style"], "-T", f"reference={REFERENCE}")
    assert result.stderr == ""


@pytest.mark.parametrize("group", NO_STYLE_GROUPS)
def test_group_without_style_refuses_reference(run, group):
    _assert_usage_error(run(group, MODEL, JUDGE, REFERENCE), "take no reference model")


@pytest.mark.parametrize("group", list(GROUPS))
def test_uncensorship_is_in_no_group(run, group):
    result = run(group, MODEL, JUDGE, *([REFERENCE] if group in STYLE_GROUPS else []))
    assert result.returncode == 0, result.stderr
    assert all("dna_eval/uncensorship" not in call for call in result.calls)


# --- single tasks ---------------------------------------------------------------------------------

@pytest.mark.parametrize("task", SINGLE_TASKS)
def test_single_task(run, task):
    result = run(task, MODEL, JUDGE)
    assert result.returncode == 0, result.stderr
    assert result.calls == [_call([task])]
    assert result.stderr == ""


@pytest.mark.parametrize("task", ["performance", "kmmlu_pro", "haerae_vision", "uncensorship"])
def test_single_task_refuses_reference(run, task):
    _assert_usage_error(run(task, MODEL, JUDGE, REFERENCE), "only style takes a reference model")


def test_style_with_reference(run):
    result = run("style", MODEL, JUDGE, REFERENCE)
    assert result.returncode == 0, result.stderr
    assert result.calls == [_call(["style"], "-T", f"reference={REFERENCE}")]


def test_style_needs_reference(run):
    _assert_usage_error(run("style", MODEL, JUDGE), "style needs a reference model")


# --- argument errors ------------------------------------------------------------------------------

@pytest.mark.parametrize("task", [
    "nonexistent", "Performance", "dna_eval/performance", "",
    # several names in one argument must not pass as one known task
    "performance hallucination", "style kmmlu_pro", "kmmmu uncensorship",
])
def test_unknown_task(run, task):
    _assert_usage_error(run(task, MODEL, JUDGE), f"unknown task: {task}")


@pytest.mark.parametrize("args", [(), ("performance",), ("performance", MODEL)], ids=["0", "1", "2"])
def test_too_few_arguments_print_usage(run, args):
    result = run(*args)
    assert result.returncode == 2
    assert result.stdout.startswith("Run DNA evaluation suite") and USAGE in result.stdout
    assert result.calls == []


# --- options --------------------------------------------------------------------------------------

OPTIONS = ["-T", "thinking=false", "--log-dir", "logs/two words", "-T", "pattern=*", "--max-connections", "16"]


@pytest.mark.parametrize("group,reference", [
    ("dna_eval", REFERENCE), ("dna_eval", None), ("all", REFERENCE), ("text_only", REFERENCE),
    ("benchmarks", None), ("vision", None),
])
def test_options_are_forwarded_to_every_group_call(run, group, reference):
    result = run(group, MODEL, JUDGE, *([reference] if reference else []), *OPTIONS)
    assert result.returncode == 0, result.stderr
    assert result.calls == _group_calls(group, reference, OPTIONS)
    assert all(call[-len(OPTIONS):] == OPTIONS for call in result.calls)


@pytest.mark.parametrize("task,reference", [("performance", None), ("uncensorship", None), ("style", REFERENCE)])
def test_options_are_forwarded_to_a_single_task(run, task, reference):
    result = run(task, MODEL, JUDGE, *([reference] if reference else []), *OPTIONS)
    assert result.returncode == 0, result.stderr
    style = ["-T", f"reference={reference}"] if task == "style" else []
    assert result.calls == [_call([task], *style, *OPTIONS)]


def test_an_option_in_place_of_the_reference_is_an_option(run):
    result = run("dna_eval", MODEL, JUDGE, "-T", "thinking=false")
    assert result.returncode == 0, result.stderr
    assert result.calls == [_call(SUITE, "-T", "thinking=false")]
    assert STYLE_SKIPPED in result.stderr


def test_arguments_pass_verbatim(run):
    model, judge, reference = "vllm/my model", "openai-api/judge/a $JUDGE", "ref *"
    result = run("dna_eval", model, judge, reference, "--log-dir", "logs/$HOME *")
    assert result.returncode == 0, result.stderr
    assert result.calls == [
        _call(SUITE, "--log-dir", "logs/$HOME *", model=model, judge=judge),
        _call(["style"], "-T", f"reference={reference}", "--log-dir", "logs/$HOME *", model=model, judge=judge),
    ]


# --- failures -------------------------------------------------------------------------------------

def test_failing_inspect_stops_before_style(run):
    result = run("dna_eval", MODEL, JUDGE, REFERENCE, inspect_exit=3)
    assert result.returncode == 3
    assert result.calls == [_call(SUITE)]


def test_failing_inspect_fails_a_single_task(run):
    result = run("performance", MODEL, JUDGE, inspect_exit=1)
    assert result.returncode == 1
    assert result.calls == [_call(["performance"])]


# --- run.sh, these lists and the package name the same tasks --------------------------------------

def test_task_lists_match_the_registered_tasks():
    assert set(SINGLE_TASKS) | {"style"} == REGISTERED
    assert len(SINGLE_TASKS) + 1 == len(REGISTERED)       # no task listed twice


@pytest.mark.parametrize("task", sorted(REGISTERED))
def test_every_registered_task_runs_by_name(run, task):
    style = task == "style"
    result = run(task, MODEL, JUDGE, *([REFERENCE] if style else []))
    assert result.returncode == 0, result.stderr
    assert result.calls == [_call([task], *(["-T", f"reference={REFERENCE}"] if style else []))]


def test_all_runs_every_registered_task_but_uncensorship(run):
    result = run("all", MODEL, JUDGE, REFERENCE)
    run_tasks = [arg.removeprefix("dna_eval/") for call in result.calls for arg in call if arg.startswith("dna_eval/")]
    assert len(run_tasks) == len(set(run_tasks))
    assert set(run_tasks) == REGISTERED - set(NAMED_ONLY)


def test_usage_names_every_registered_task(run):
    words = set(run().stdout.replace(",", " ").split())
    assert REGISTERED <= words, f"run.sh usage does not name: {sorted(REGISTERED - words)}"
