"""SciCode: 65 research-level scientific coding problems, each written and tested sub-step by sub-step."""
from __future__ import annotations

from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, hf_dataset

import dna_eval.vendor.scicode as vendored
from dna_eval.metrics.sub_step_accuracy import sub_step_accuracy
from dna_eval.scorers.scicode_tests import SCICODE_INSTALL, scicode_tests
from dna_eval.solvers.scicode_steps import scicode_steps
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics
from dna_eval.utils.cache import cache_root, fetch_pinned
from dna_eval.utils.optional import require

# SciCode1/SciCode commit this task is pinned to (Apache-2.0).
DATASET_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"

# The numeric test targets (``test_data.h5``, 1,049,345,865 bytes). The authors publish them in
# a Google Drive folder, which cannot be fetched by checksum; this Hub mirror (Apache-2.0, commit
# below) holds a file with the SHA-256 that other Hub copies of the file carry too.
H5_URL = ("https://huggingface.co/datasets/xantheocracy/scicode-mirror/resolve/"
          "2f903a64a1c4eb62c7c649b0caf7ecb2c0217590/test_data.h5")
H5_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"

# The sub-steps the benchmark neither generates nor tests, with the file of gold code its
# generation code shows the model in their place.
SKIPPED = {"13.6": "13.6.txt", "62.1": "62.1.txt", "76.3": "76.3.txt"}

# Python 3.11 with numpy, scipy, sympy, h5py and matplotlib, no network (a Docker Compose file; ``--sandbox``
# swaps the provider).
SANDBOX = ("docker", str(Path(__file__).with_name("scicode_sandbox") / "compose.yaml"))

STEPS_KEY = "scicode_steps"     # state.store: the solver's per-sub-step entries, read by the scorer


def record_to_sample(record: dict) -> Sample:
    """One problem. ``input`` is the problem's overall description, for the log viewer only: the
    solver builds the prompt of every sub-step itself from ``metadata``. Each sub-step keeps what
    the prompts and the tests need; a skipped sub-step also carries its gold code
    (``skipped_code``). The dataset's own solutions (``ground_truth_code``) are left out."""
    sub_steps = []
    for step in record["sub_steps"]:
        gold = SKIPPED.get(step["step_number"])
        sub_steps.append({
            **{k: step[k] for k in ("step_number", "step_description_prompt", "step_background",
                                    "function_header", "return_line", "test_cases")},
            "skipped_code": None if gold is None else (Path(vendored.__file__).parent / gold).read_text(encoding="utf-8"),
        })
    return Sample(
        id=record["problem_id"],
        input=record["problem_description_main"],
        target="",
        metadata={"problem_name": record["problem_name"], "required_dependencies": record["required_dependencies"],
                  "sub_steps": sub_steps},
    )


def h5_file(sha256: str = H5_SHA256) -> Path:
    """The test targets, downloaded once into the cache directory and named by their checksum
    (about 1 GB; the whole file is read to check it each time this is called)."""
    return fetch_pinned(H5_URL, cache_root() / "benchmarks" / "scicode" / f"{sha256}.h5", sha256)


@task
def scicode(thinking: bool = True, epochs: int = 3, with_background: bool = False,
            revision: str = DATASET_REVISION) -> Task:
    """SciCode (Tian et al., 2024), test split: 65 problems in 288 evaluated sub-steps (291, minus
    three the benchmark skips). The model writes each sub-step's function in turn; the problem
    scores 1 when every evaluated sub-step passes its numeric unit tests. The headline is
    ``accuracy``, the share of problems solved, averaged over ``epochs`` runs; ``sub_step_accuracy``
    is the share of sub-steps passed, pooled over runs.

    Source followed: the benchmark's released code (``scicode-bench/SciCode``, commit
    e3158ea: ``eval/scripts/gencode.py`` for prompts and code extraction, ``eval/inspect_ai/scicode.py``
    and ``eval/scripts/test_generated_code.py`` for testing) and its data (``SciCode1/SciCode`` on
    the Hub, pinned by ``revision``; ``test_data.h5`` pinned by checksum). Each sub-step is a fresh
    one-turn prompt, as there: the earlier steps' descriptions and the code the model gave for
    them, then the step's description and function header, in the benchmark's template
    (``solvers/scicode_steps.py``). ``with_background=False`` (the benchmark's
    default) has the model write the scientific background as a comment before its code;
    ``True`` puts the dataset's background text in the step descriptions instead. The code is
    the first ```python block of the final content with import lines blanked. Each sub-step is
    tested in its own process, with a 1800 s limit, and passes when its script exits with
    status 0 (``scorers/scicode_tests.py``).

    Deviations:

    - thinking is on and no ``max_tokens`` is set, with sampling from the served model's
      ``generation_config`` (the benchmark's scripts sample at temperature 0 and cap Claude's
      output at 4096 tokens); only the final content is read for code, never the reasoning;
    - a truncated reply fails its sub-step without being tested (the benchmark has no such
      case), and ``answer_diagnostics`` counts truncated problems and problems with a reply
      that held no code block (``format_failure``);
    - the benchmark's own scorer in ``eval/inspect_ai`` counts problems 13, 62 and 76 against a
      denominator that includes their skipped sub-step, so they cannot pass; here a problem passes
      when all of its evaluated sub-steps pass, as in ``test_generated_code.py``;
    - the tests run in a sandbox (a Python 3.11 image with numpy 1.26, scipy 1.13, sympy, h5py
      and matplotlib; no network) instead of a local subprocess, and read their targets from an HDF5 file
      holding only the problem's groups of ``test_data.h5``, copied unchanged and loaded by the
      benchmark's own ``process_hdf5_to_tuple``; the test cases are as in the dataset. The
      targets come from a Hub mirror instead of the authors' Google Drive folder;
    - a generation call that fails is a sample error to retry, not replaced by a dummy reply as
      in the benchmark's Inspect task;
    - for the skipped sub-steps 13.6 and 62.1 the gold code shown to the model is reduced, as the
      benchmark's lookup does it (``extract_function_name`` returns ``__init__`` for a class
      header), to the first ``__init__`` method of the file, unindented; this is kept as it is;
    - ``epochs`` defaults to 3 (the benchmark scores one run; Artificial Analysis reports the average of three).

    Needs ``h5py`` on the host (``pip install 'dna-eval[scicode]'``) and a sandbox provider that
    runs ``SANDBOX`` (Docker by default; ``--sandbox`` selects another, whose image needs the
    packages above). Downloads the 1 GB test file on first use.
    """
    dataset = hf_dataset("SciCode1/SciCode", split="test", revision=revision, sample_fields=record_to_sample)
    require("h5py", install=SCICODE_INSTALL)      # before the 1 GB download
    return Task(
        dataset=dataset,
        solver=scicode_steps(with_background=with_background, key=STEPS_KEY),
        scorer=scicode_tests(h5_file(), key=STEPS_KEY),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
        metrics=[*benchmark_metrics(), sub_step_accuracy()],
        sandbox=SANDBOX,
    )
