"""``scicode_tests``: run SciCode's unit tests for every sub-step of a problem in the sandbox."""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import OutputLimitExceededError, sandbox

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.sub_step_accuracy import STEPS_PASSED, STEPS_TOTAL, sub_step_accuracy
from dna_eval.utils.optional import require

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "scicode"
WORKDIR = "scicode_tests"       # directory in the sandbox that holds the scripts, relative to its working directory
TARGETS = "targets.h5"          # the targets of the problem's sub-steps, inside WORKDIR
TIMEOUT_EXIT = 124              # exit status of coreutils ``timeout``, which some sandbox providers use
SCICODE_INSTALL = 'pip install "dna-eval[scicode]"'   # h5py, to read the test targets on the host

# Test-script text, as the benchmark's evaluator writes it (``eval/inspect_ai/scicode.py``,
# ``ScicodeEvaluator.test_code``; scicode-bench/SciCode, Apache-2.0; see THIRD_PARTY_NOTICES.md).
IMPORT_TARGETS = "\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n\n"


def assemble_script(code: str, step_id: str, test_cases: list[str], targets_file: str) -> str:
    """The script that tests one sub-step: ``code`` (the dependencies, the earlier sub-steps and
    this one), then the loading of the step's targets from ``targets_file``, then each test case
    preceded by ``target = targets[i]``. Exit status 0 means every test passed.

    Same text as the benchmark's evaluator writes, with ``targets_file`` as the HDF5 file.
    """
    parts = [code, IMPORT_TARGETS, f"targets = process_hdf5_to_tuple('{step_id}', {len(test_cases)}, '{targets_file}')\n"]
    for i, test in enumerate(test_cases):
        parts.append(f"target = targets[{i}]\n\n")
        parts.extend(line + "\n" for line in test.split("\n"))
    return "".join(parts)


def extract_targets(h5_path: str | Path, step_ids: list[str]) -> bytes:
    """An HDF5 file holding only the groups of ``step_ids``, copied unchanged from ``h5_path``.

    The full test file is about 1 GB; a problem's sub-steps need a few megabytes of it.
    """
    require("h5py", install=SCICODE_INSTALL)
    import h5py

    buffer = io.BytesIO()
    with h5py.File(str(h5_path), "r") as source, h5py.File(buffer, "w") as dest:
        for step_id in step_ids:
            if step_id not in source:
                raise KeyError(f"{h5_path} has no group {step_id!r}: not SciCode's test_data.h5")
            source.copy(source[step_id], dest, name=step_id)
    return buffer.getvalue()


@scorer(metrics=[accuracy(), stderr(), sub_step_accuracy(), answer_diagnostics()])
def scicode_tests(h5_path: str | Path, timeout: int = 1800, key: str = "scicode_steps",
                  python: str = "python") -> Scorer:
    """Test the code of every sub-step of a SciCode problem; the problem scores 1.0 when all of
    its evaluated sub-steps pass, else 0.0.

    ``state.store[key]`` holds the entries ``scicode_steps`` wrote, ``state.metadata["sub_steps"]``
    the test cases. Each sub-step that was written, was not truncated and is not one the
    benchmark skips is tested in its own process, in the sandbox, with ``python`` and a limit of
    ``timeout`` seconds (the benchmark's 1800): the script is the step's saved code followed by
    its test cases (``assemble_script``), and exit status 0 is a pass. The sub-steps run one after
    another. The targets the tests compare with are read from ``h5_path`` (SciCode's
    ``test_data.h5``) on the host and only the problem's own groups are sent to the sandbox, next to a copy
    of the benchmark's comparison helpers (``scicode.compare.cmp``) and HDF5 reader
    (``scicode.parse.parse``); the sandbox needs Python with numpy, scipy, sympy, h5py and
    matplotlib (some problems import it), and no network. A truncated sub-step fails without being run.

    ``metadata`` holds ``steps_passed`` and ``steps_total`` (``sub_step_accuracy``), ``steps``
    (per sub-step: ``step_number``, ``status`` (``pass``, ``fail``, ``timeout``, ``output_limit``,
    ``truncated``), and for a failure the end of its stderr), ``truncated`` (any sub-step was
    cut off) and ``extracted`` (``None`` when a sub-step's reply held no code block, which the
    benchmark's code then takes as the code as a whole).
    """
    require("h5py", install=SCICODE_INSTALL)

    async def score(state: TaskState, target: Target) -> Score:
        entries: list[dict[str, Any]] | None = state.store.get(key)
        if entries is None:
            raise RuntimeError(f"state.store[{key!r}] is empty: run the scicode_steps solver first")
        sub_steps = state.metadata["sub_steps"]
        if len(entries) != len(sub_steps):
            raise RuntimeError(f"{len(entries)} generated sub-steps for a problem with {len(sub_steps)}")

        evaluated = [(e, s) for e, s in zip(entries, sub_steps) if not e["skipped"]]
        runnable = [e["step_number"] for e, _ in evaluated if not e["truncated"]]
        results: dict[str, dict[str, Any]] = {}
        if runnable:
            targets = await asyncio.to_thread(extract_targets, h5_path, runnable)
            results = await run_tests(evaluated, targets, timeout, python)

        steps = [{"step_number": e["step_number"],
                  **(results[e["step_number"]] if e["step_number"] in results else {"status": "truncated"})}
                 for e, _ in evaluated]
        passed = sum(1 for s in steps if s["status"] == "pass")
        truncated = any(e["truncated"] for e, _ in evaluated)
        fenced = all(e["fenced"] for e, _ in evaluated)
        correct = passed == len(steps)
        return Score(
            value=1.0 if correct else 0.0,
            answer=f"{passed}/{len(steps)} sub-steps",
            explanation=", ".join(f"{s['step_number']}:{s['status']}" for s in steps),
            metadata={STEPS_PASSED: passed, STEPS_TOTAL: len(steps), "steps": steps, TRUNCATED: truncated,
                      EXTRACTED: f"{len(steps)} code blocks" if fenced else None},
        )

    return score


async def run_tests(evaluated: list[tuple[dict, dict]], targets: bytes, timeout: int, python: str) -> dict[str, dict]:
    """Run the test script of each ``(entry, sub_step)`` pair that was not truncated in the sandbox
    and return ``{step_number: {"status": ..., "stderr": ...}}``."""
    sb = sandbox()
    await sb.write_file(f"{WORKDIR}/{TARGETS}", targets)
    for package in ("scicode", "scicode/compare", "scicode/parse"):
        await sb.write_file(f"{WORKDIR}/{package}/__init__.py", "")
    await sb.write_file(f"{WORKDIR}/scicode/compare/cmp.py", (VENDOR / "cmp.py").read_text(encoding="utf-8"))
    await sb.write_file(f"{WORKDIR}/scicode/parse/parse.py", (VENDOR / "parse.py").read_text(encoding="utf-8"))
    results: dict[str, dict] = {}
    try:
        for entry, sub_step in evaluated:
            if entry["truncated"]:
                continue
            step_id = entry["step_number"]
            script = assemble_script(f"{entry['previous_code']}\n{entry['code']}", step_id, sub_step["test_cases"], TARGETS)
            await sb.write_file(f"{WORKDIR}/{step_id}.py", script)
            try:
                result = await sb.exec([python, f"{step_id}.py"], cwd=WORKDIR, timeout=timeout, timeout_retry=False)
            except TimeoutError:
                results[step_id] = {"status": "timeout"}
            except OutputLimitExceededError:
                results[step_id] = {"status": "output_limit"}
            else:
                if result.returncode == 0:
                    results[step_id] = {"status": "pass"}
                else:
                    status = "timeout" if result.returncode == TIMEOUT_EXIT else "fail"
                    results[step_id] = {"status": status, "stderr": result.stderr[-1000:]}
    finally:
        try:
            await sb.exec(["rm", "-rf", WORKDIR], timeout=60)
        except Exception:
            pass        # the sandbox is discarded after the sample; a failed clean-up changes nothing
    return results
