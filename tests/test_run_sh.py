"""The ``inspect eval`` commands ``run.sh`` builds.

``run.sh`` is run with a fake ``inspect`` first on ``PATH`` that only records its arguments,
so nothing here evaluates a model or reaches an endpoint.

- A group runs its tasks in one ``inspect eval``; ``style`` runs in a second one, with the
  reference model, and is the only task skipped when no reference model is given.
- The groups cover the tasks the package registers (``_registry.py``), so a task added to one
  and not the other fails here. ``uncensorship`` is in no group: it runs only when named.
- Model, judge, reference and extra options reach ``inspect eval`` unchanged.
- A wrong call (unknown task, missing or misplaced reference model) prints the usage, exits 2
  and runs nothing.
"""
import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "run.sh"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None or os.name == "nt", reason="run.sh needs bash")

MODEL = "vllm/model-under-test"
JUDGE = "openai-api/judge/judge-model"
REFERENCE = "reference-model"

SUITE = ["performance", "hallucination", "persona", "degeneration", "style"]
BENCHMARKS = ["kmmlu_pro", "kmmlu_redux", "kmmlu_hard", "click", "kobalt", "hrm8k_ksm", "haerae_bench",
              "kosimpleqa", "kormedmcqa", "kbl", "haerae_vision", "kmmmu"]
VISION = ["haerae_vision", "kmmmu"]
NAMED_ONLY = ["uncensorship"]
GROUPS = {
    "dna_eval": SUITE,
    "benchmarks": BENCHMARKS,
    "all": SUITE + BENCHMARKS,
    "text_only": [t for t in SUITE + BENCHMARKS if t not in VISION],
    "vision": VISION,
}
NO_REFERENCE_GROUPS = ["benchmarks", "vision"]      # groups without style: a reference model is an error

# Records one invocation as its NUL-separated arguments followed by a record separator.
FAKE_INSPECT = """#!{bash}
{{ printf '%s\\0' "$@"; printf '\\036'; }} >> "$FAKE_INSPECT_LOG"
exit "${{FAKE_INSPECT_EXIT:-0}}"
"""


def _registered_tasks() -> tuple[set[str], set[str]]:
    """(suite tasks, benchmark tasks) the package registers, read from ``_registry.py``'s imports."""
    tree = ast.parse((ROOT / "src" / "dna_eval" / "_registry.py").read_text(encoding="utf-8"))
    suite, benchmarks = set(), set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("dna_eval.tasks."):
            into = benchmarks if node.module.startswith("dna_eval.tasks.benchmarks.") else suite
            into.update(alias.name for alias in node.names)
    return suite, benchmarks


@pytest.fixture
def run(tmp_path):
    """Run ``run.sh`` with the given arguments; returns (process, recorded ``inspect`` calls)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "inspect"
    fake.write_text(FAKE_INSPECT.format(bash=BASH), encoding="utf-8")
    fake.chmod(0o755)
    log = tmp_path / "inspect-calls"
    cwd = tmp_path / "cwd"          # empty: no .env, no logs/
    cwd.mkdir()

    def _run(*args: str, inspect_exit: int = 0):
        # A minimal environment: the fake first on PATH, none of the caller's endpoints or keys.
        env = {"PATH": os.pathsep.join([str(bin_dir), str(Path(BASH).parent), "/usr/bin", "/bin"]),
               "FAKE_INSPECT_LOG": str(log), "FAKE_INSPECT_EXIT": str(inspect_exit)}
        proc = subprocess.run([BASH, str(RUN_SH), *args], cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=30)
        records = log.read_text(encoding="utf-8").split("\x1e")[:-1] if log.exists() else []
        log.unlink(missing_ok=True)
        return proc, [r.split("\0")[:-1] for r in records]

    return _run


def _command(tasks: list[str], *options: str) -> list[str]:
    return ["eval", *[f"dna_eval/{t}" for t in tasks], "--model", MODEL, "--model-role", f"grader={JUDGE}",
            *options]


def _style_command(*options: str) -> list[str]:
    return _command(["style"], "-T", f"reference={REFERENCE}", *options)


# --- groups ---------------------------------------------------------------------------------

@pytest.mark.parametrize("group", GROUPS)
def test_group_without_reference_skips_only_style(run, group):
    proc, calls = run(group, MODEL, JUDGE)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_command([t for t in GROUPS[group] if t != "style"])]
    assert ("style skipped" in proc.stderr) == ("style" in GROUPS[group])


@pytest.mark.parametrize("group", [g for g in GROUPS if g not in NO_REFERENCE_GROUPS])
def test_group_with_reference_runs_style_separately(run, group):
    proc, calls = run(group, MODEL, JUDGE, REFERENCE)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_command([t for t in GROUPS[group] if t != "style"]), _style_command()]
    assert "style skipped" not in proc.stderr


@pytest.mark.parametrize("group", NO_REFERENCE_GROUPS)
def test_group_without_style_rejects_reference(run, group):
    proc, calls = run(group, MODEL, JUDGE, REFERENCE)
    assert proc.returncode == 2
    assert "no reference model" in proc.stderr
    assert calls == []


def test_groups_cover_the_registered_tasks(run):
    """A task registered with Inspect is in its group, and a group names no unregistered task."""
    suite, benchmarks = _registered_tasks()
    assert suite and benchmarks

    def tasks_of(group: str) -> set[str]:
        proc, calls = run(group, MODEL, JUDGE, *([] if group in NO_REFERENCE_GROUPS else [REFERENCE]))
        assert proc.returncode == 0, proc.stderr
        names = [a for call in calls for a in call if a.startswith("dna_eval/")]
        assert len(names) == len(set(names)), f"{group} runs a task twice: {names}"
        return {n.removeprefix("dna_eval/") for n in names}

    assert tasks_of("dna_eval") == suite - set(NAMED_ONLY)
    assert tasks_of("benchmarks") == benchmarks
    assert tasks_of("all") == (suite | benchmarks) - set(NAMED_ONLY)
    assert tasks_of("vision") == set(VISION) <= benchmarks
    assert tasks_of("text_only") == (suite | benchmarks) - set(NAMED_ONLY) - set(VISION)


def test_usage_names_every_registered_task(run):
    suite, benchmarks = _registered_tasks()
    proc, calls = run()
    assert proc.returncode == 2 and calls == []
    missing = [t for t in sorted(suite | benchmarks) if t not in proc.stdout]
    assert not missing, f"run.sh usage does not mention: {missing}"
    assert not proc.stdout.startswith("#") and "set -euo" not in proc.stdout     # the header comment, only


# --- single tasks ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", [t for t in SUITE + BENCHMARKS + NAMED_ONLY if t != "style"])
def test_single_task(run, task):
    proc, calls = run(task, MODEL, JUDGE)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_command([task])]
    assert proc.stderr == ""


def test_uncensorship_runs_only_when_named(run):
    for group in GROUPS:
        _, calls = run(group, MODEL, JUDGE, *([] if group in NO_REFERENCE_GROUPS else [REFERENCE]))
        assert calls and not any("dna_eval/uncensorship" in call for call in calls), group
    _, calls = run("uncensorship", MODEL, JUDGE)
    assert calls == [_command(["uncensorship"])]


def test_style_with_reference(run):
    proc, calls = run("style", MODEL, JUDGE, REFERENCE)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_style_command()]


@pytest.mark.parametrize("options", [[], ["-T", "thinking=false"]], ids=["bare", "options"])
def test_style_without_reference_is_an_error(run, options):
    proc, calls = run("style", MODEL, JUDGE, *options)
    assert proc.returncode == 2
    assert "style needs a reference model" in proc.stderr
    assert calls == []


@pytest.mark.parametrize("task", ["performance", "kbl", "uncensorship"])
def test_other_task_rejects_reference(run, task):
    proc, calls = run(task, MODEL, JUDGE, REFERENCE)
    assert proc.returncode == 2
    assert "only style takes a reference model" in proc.stderr
    assert calls == []


# --- wrong calls ----------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["no_such_task", "dna_eval/performance", "Performance", "perf", ""])
def test_unknown_task(run, task):
    proc, calls = run(task, MODEL, JUDGE)
    assert proc.returncode == 2
    assert f"unknown task: {task}" in proc.stderr
    assert "bash run.sh <task> <model> <judge-model>" in proc.stdout       # the usage follows
    assert calls == []


@pytest.mark.parametrize("args", [[], ["performance"], ["performance", MODEL]], ids=["0", "1", "2"])
def test_too_few_arguments_print_usage(run, args):
    proc, calls = run(*args)
    assert proc.returncode == 2
    assert "bash run.sh <task> <model> <judge-model>" in proc.stdout
    assert calls == []


# --- options --------------------------------------------------------------------------------

OPTIONS = ["-T", "thinking=false", "--log-dir", "logs/a dir with spaces", "--max-connections", "16",
           "--limit", "5", "-M", "note=it's * $HOME \"quoted\"", ""]


def test_options_pass_through_single_task(run):
    proc, calls = run("performance", MODEL, JUDGE, *OPTIONS)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_command(["performance"], *OPTIONS)]


def test_options_pass_through_style(run):
    _, calls = run("style", MODEL, JUDGE, REFERENCE, *OPTIONS)
    assert calls == [_style_command(*OPTIONS)]


def test_options_reach_both_runs_of_a_group(run):
    _, calls = run("dna_eval", MODEL, JUDGE, REFERENCE, *OPTIONS)
    assert calls == [_command([t for t in SUITE if t != "style"], *OPTIONS), _style_command(*OPTIONS)]


def test_option_after_judge_is_not_a_reference(run):
    """The fourth argument is the reference model only when it does not start with a dash."""
    proc, calls = run("dna_eval", MODEL, JUDGE, "--limit", "5")
    assert calls == [_command([t for t in SUITE if t != "style"], "--limit", "5")]
    assert "style skipped" in proc.stderr


def test_no_options_are_added(run):
    """run.sh sets no defaults of its own (concurrency, log directory, ...): those stay Inspect's."""
    _, calls = run("performance", MODEL, JUDGE)
    assert calls == [["eval", "dna_eval/performance", "--model", MODEL, "--model-role", f"grader={JUDGE}"]]


def test_model_names_pass_unchanged(run):
    model, judge, reference = "vllm/org/my model:v1", "openai-api/judge/org/j*dge", "org/ref model"
    _, calls = run("style", model, judge, reference)
    assert calls == [["eval", "dna_eval/style", "--model", model, "--model-role", f"grader={judge}",
                      "-T", f"reference={reference}"]]


# --- failures -------------------------------------------------------------------------------

def test_inspect_failure_is_reported(run):
    proc, calls = run("performance", MODEL, JUDGE, inspect_exit=3)
    assert proc.returncode == 3
    assert len(calls) == 1


def test_group_stops_when_the_first_run_fails(run):
    proc, calls = run("dna_eval", MODEL, JUDGE, REFERENCE, inspect_exit=1)
    assert proc.returncode == 1
    assert calls == [_command([t for t in SUITE if t != "style"])]      # style is not run after a failure
