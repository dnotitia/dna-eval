"""KMMLU-Redux: 2,587 multiple-choice questions from Korean National Technical Qualification exams."""
from __future__ import annotations

import json

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate

from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics
from dna_eval.tasks.benchmarks.kmmlu_pro import ANSWER_PATTERN, LETTERS, build_prompt

REPO = "LGAI-EXAONE/KMMLU-Redux"     # gated: needs an access grant and a Hub token
DATA_FILE = "data/kmmlu_redux.jsonl"
# Commit this task is pinned to.
DATASET_REVISION = "5ae54a2048e4e2754ef3fb3b0c19c8ebe8111eb4"


def record_to_sample(record: dict, index: int) -> Sample:
    """One question: ``solution`` is the 1-based number of the correct option."""
    return Sample(
        id=index,
        input=build_prompt(record["question"], record["options"]),
        target=LETTERS[int(record["solution"]) - 1],
        metadata={"category": record["category"], "license_name": record["license_name"]},
    )


def load_samples(revision: str) -> list[Sample]:
    """The benchmark file at ``revision``; sample ids are row numbers."""
    path = hf_hub_download(REPO, DATA_FILE, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [record_to_sample(r, i) for i, r in enumerate(rows)]


@task
def kmmlu_redux(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """KMMLU-Redux, 2,587 questions x ``epochs`` runs; accuracy overall and by category.

    KMMLU-Redux and KMMLU-Pro come from the same paper (Hong et al., 2025, arXiv:2507.08924),
    which evaluates both with the same Korean zero-shot prompt (Appendix E.1, Figure 10) and
    reports plain accuracy, broken down by the 14 industrial categories. Official evaluation
    code is published for KMMLU-Pro only (LG-AI-EXAONE/KMMLU-Pro); this task uses that code's
    four-option prompt and ``정답:`` answer pattern, imported from ``kmmlu_pro``.

    The dataset is gated on the Hub: request access and provide a token (``HF_TOKEN``).

    Deviations, all for reasoning models:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the paper decodes greedily for non-reasoning models, and at temperature 0.6, top-p 0.95
      for reasoning models);
    - the answer is read from the final content only, and the last match is used, as the
      official code does; the paper also lists fallback patterns, which the official code
      does not use;
    - a truncated reply scores 0 even if it contains an answer line;
    - the question file is read directly from ``data/kmmlu_redux.jsonl`` at the pinned commit
      instead of through ``datasets.load_dataset``.
    """
    return Task(
        dataset=MemoryDataset(load_samples(revision), name="kmmlu_redux"),
        solver=generate(),
        scorer=letter_choice([ANSWER_PATTERN], letters=LETTERS),
        metrics=benchmark_metrics("category"),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
