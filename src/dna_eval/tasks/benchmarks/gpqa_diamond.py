"""GPQA-Diamond: 198 graduate-level multiple-choice science questions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate

from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.solvers.shuffled_choices import CORRECT_LETTER_KEY, shuffled_choices
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config
from dna_eval.utils.cache import cache_root, fetch_pinned

# The question file the reference implementation reads, pinned by checksum.
DATA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
DATA_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"

PROMPT = ("Answer the following multiple choice question. The last line of your response should be of "
          "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
          "Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")
ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def data_file(url: str = DATA_URL, sha256: str = DATA_SHA256) -> Path:
    """The question file, downloaded once into the cache directory and named by its checksum;
    a copy whose SHA-256 differs from ``sha256`` is rejected (``-T sha256=`` accepts a new one)."""
    return fetch_pinned(url, cache_root() / "benchmarks" / "gpqa_diamond" / f"{sha256}.csv", sha256)


def load_samples(path: Path) -> list[Sample]:
    """One sample per question. The options are stored correct-first in ``metadata["choices"]``;
    ``shuffled_choices`` orders them per run."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [Sample(
        id=row["Record ID"],
        input=row["Question"],
        target=row["Correct Answer"],
        metadata={
            "choices": [row["Correct Answer"], row["Incorrect Answer 1"],
                        row["Incorrect Answer 2"], row["Incorrect Answer 3"]],
            "domain": row["High-level domain"],
            "subdomain": row["Subdomain"],
        },
    ) for _, row in df.iterrows()]


@task
def gpqa_diamond(thinking: bool = True, epochs: int = 8, seed: int = 0, sha256: str = DATA_SHA256) -> Task:
    """GPQA-Diamond, 198 questions x ``epochs`` runs; the score is accuracy averaged over runs.

    Follows the reference implementation in openai/simple-evals (``gpqa_eval.py``): the same
    question file, the same prompt, the same answer pattern, one user turn, options
    shuffled before they are shown. The data is that implementation's CSV, fetched from its
    URL and checked against ``sha256``; it is the Diamond subset of ``Idavidrein/gpqa``
    (the same bytes), so no Hub access is needed.

    Deviations, all for reasoning models:

    - thinking is on and no ``max_tokens`` is set; sampling comes from the served model's
      ``generation_config`` (the reference samples at temperature 0.5);
    - the answer is the *last* ``Answer: X`` in the final content, not the first, so a
      response that writes such a line mid-way is not scored on it;
    - a truncated response scores 0 even if it contains an answer line (the reference has no
      such case);
    - the option order is a function of ``seed``, the question and the epoch (the reference
      draws one permutation per repeat from a single random stream), so a rerun is reproducible
      and every epoch sees a different order;
    - ``epochs`` defaults to 8 (the reference's default is 4).
    """
    return Task(
        dataset=MemoryDataset(load_samples(data_file(sha256=sha256)), name="gpqa_diamond"),
        solver=[shuffled_choices(PROMPT, seed=seed), generate()],
        scorer=letter_choice([ANSWER_PATTERN], letters="ABCD", target_key=CORRECT_LETTER_KEY),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
