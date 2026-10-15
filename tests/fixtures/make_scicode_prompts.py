"""Regenerate ``scicode/official_prompts.json``: what SciCode's own code makes of three problems.

Needs a clone of https://github.com/scicode-bench/SciCode and the packages its modules import
(``h5py``, ``scipy``, ``sympy``, ``datasets``, ``numpy``, ``inspect_ai``), none of them
dependencies of the core package. Run from the repository root::

    python tests/fixtures/make_scicode_prompts.py --upstream /path/to/SciCode

The official generation code (``eval/scripts/gencode.py``, class ``Gencode``) is run unchanged
for problems 14, 62 and 76 (62 and 76 hold sub-steps the benchmark skips) with a model function
that returns the canned replies below, with and without background. The official evaluator
(``eval/inspect_ai/scicode.py``, ``ScicodeEvaluator.test_code``) assembles the test scripts, with
``subprocess.run`` replaced by a recorder so nothing is executed. The fixture records the
problem data the code was given, the replies, the prompts it built, the code files it wrote and
the test scripts it assembled; ``tests/test_scicode.py`` compares our code with them.

SciCode's modules read ``eval/data`` relative to the working directory and import every model
SDK; the SDKs are replaced by stubs here because only ``extract_python_script`` is used.
"""
import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

DATASET = "SciCode1/SciCode"
DATASET_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"
H5_PLACEHOLDER = "TARGETS.h5"          # stands for the file the test scripts read targets from
PROBLEMS = ["14", "62", "76"]
OUT = Path(__file__).with_name("scicode") / "official_prompts.json"

FENCED = ("Here is the code.\n```python\nimport numpy as np\nfrom scipy import linalg as la\n"
          "# step {n}\ndef f_{n}(x):\n    return x + {n}\n```\nDone.")
PLAIN_FENCE = "Answer:\n```\nimport math\ndef g_{n}(x):\n    return math.floor(x) * {n}\n```\n"
UNFENCED = "import numpy as np\ndef h_{n}(x):\n    return x - {n}\n"


def canned_reply(problem_id: str, step_number: str) -> str:
    """The model's reply to one sub-step: mostly a ```python block, one plain ``` block and one
    reply without any block per problem that has enough steps."""
    n = step_number.split(".")[1]
    if problem_id == "14" and n == "2" or problem_id == "76" and n == "4":
        return PLAIN_FENCE.format(n=n)
    if problem_id == "62" and n == "4":
        return UNFENCED.format(n=n)
    return FENCED.format(n=n)


def stub_sdks() -> None:
    for name in ["openai", "anthropic", "google", "google.generativeai", "config", "litellm", "litellm.utils"]:
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = mock.MagicMock(__spec__=importlib.machinery.ModuleSpec(name, None))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(upstream: Path, out: Path) -> None:
    upstream = upstream.resolve()
    sys.path.insert(0, str(upstream / "src"))
    stub_sdks()
    from datasets import load_dataset
    records = {r["problem_id"]: r for r in load_dataset(DATASET, split="test", revision=DATASET_REVISION)
               if r["problem_id"] in PROBLEMS}

    os.chdir(upstream)
    gencode = load_module("scicode_gencode", upstream / "eval" / "scripts" / "gencode.py")
    os.chdir(upstream / "eval" / "inspect_ai")
    evaluator_module = load_module("scicode_inspect", upstream / "eval" / "inspect_ai" / "scicode.py")
    os.chdir(upstream)

    skipped = {("13", 5), ("62", 0), ("76", 2)}
    runs = []
    work = Path(tempfile.mkdtemp(prefix="scicode_fixture_"))
    try:
        for problem_id in PROBLEMS:
            record = records[problem_id]
            steps = record["sub_steps"]
            for with_background in (False, True):
                if with_background and problem_id == "14":
                    continue
                tag = f"{problem_id}-{with_background}"
                replies = {s["step_number"]: canned_reply(problem_id, s["step_number"])
                           for i, s in enumerate(steps) if (problem_id, i) not in skipped}
                queue = [replies[s["step_number"]] for i, s in enumerate(steps) if (problem_id, i) not in skipped]
                gen = gencode.Gencode("fixture-model", work / tag / "code", work / tag / "prompt",
                                      with_background, 0.0)
                template = gencode.BACKGOUND_PROMPT_TEMPLATE if with_background else gencode.DEFAULT_PROMPT_TEMPLATE
                with mock.patch.object(gencode, "get_model_function",
                                       lambda model, **kw: (lambda prompt: queue.pop(0))):
                    for i in range(len(steps)):
                        if (problem_id, i) in skipped:
                            continue
                        gen.generate_response_with_steps(record, i + 1, len(steps), "fixture-model", template)
                assert not queue
                bg = "with_background" if with_background else "without_background"
                prompts, codes = {}, {}
                for i, s in enumerate(steps):
                    if (problem_id, i) in skipped:
                        continue
                    sid = s["step_number"]
                    prompts[sid] = (work / tag / "prompt" / "fixture-model" / bg / f"{sid}.txt").read_text(encoding="utf-8")
                    codes[sid] = (work / tag / "code" / "fixture-model" / bg / f"{sid}.py").read_text(encoding="utf-8")

                # the official test-script assembly, over those code files
                code_dir = work / tag / "evaluator" / "generated_code" / bg
                code_dir.mkdir(parents=True)
                for sid, text in codes.items():
                    (code_dir / f"{sid}.py").write_text(text, encoding="utf-8")
                scratch = work / tag / "cwd"
                scratch.mkdir()
                os.chdir(scratch)
                kept = {}
                recorder = types.SimpleNamespace(
                    run=lambda *a, **k: None,
                    CalledProcessError=Exception, TimeoutExpired=Exception)
                keep = types.SimpleNamespace(rmtree=lambda p: kept.update(
                    {f.name: f.read_text(encoding="utf-8") for f in Path(p).iterdir()}))
                with mock.patch.object(evaluator_module, "subprocess", recorder), \
                        mock.patch.object(evaluator_module, "shutil", keep):
                    evaluator = evaluator_module.ScicodeEvaluator(
                        h5py_file=H5_PLACEHOLDER, code_dir=work / tag / "evaluator",
                        log_dir=work / tag / "logs", with_background=with_background)
                    evaluator.test_code(dict(record))
                os.chdir(upstream)
                scripts = {name.removesuffix(".py"): text for name, text in kept.items()}
                assert set(scripts) == set(codes), (set(scripts), set(codes))
                runs.append({"problem_id": problem_id, "with_background": with_background,
                             "replies": replies, "prompts": prompts, "code_files": codes,
                             "test_scripts": scripts})
    finally:
        os.chdir(upstream)
        shutil.rmtree(work, ignore_errors=True)

    problems = [{
        "problem_id": r["problem_id"],
        "required_dependencies": r["required_dependencies"],
        "sub_steps": [{k: s[k] for k in ("step_number", "step_description_prompt", "step_background",
                                         "function_header", "return_line", "test_cases")}
                      for s in r["sub_steps"]],
    } for r in records.values()]
    fixture = {
        "source": {"repository": "https://github.com/scicode-bench/SciCode",
                   "commit": _git_head(upstream), "dataset": DATASET, "dataset_revision": DATASET_REVISION,
                   "h5_placeholder": H5_PLACEHOLDER},
        "problems": problems,
        "runs": runs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(runs)} runs)")


def _git_head(path: Path) -> str:
    return (path / ".git" / "HEAD").exists() and \
        __import__("subprocess").run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", type=Path, required=True, help="a clone of scicode-bench/SciCode")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    main(args.upstream, args.out)
