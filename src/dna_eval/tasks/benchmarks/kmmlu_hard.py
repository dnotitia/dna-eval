"""KMMLU-HARD: the 4,104 KMMLU questions that proprietary models found hardest, in 45 subjects."""
from __future__ import annotations

import csv

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import HeadlineMetric
from inspect_ai.solver import generate

from dna_eval.metrics.macro_accuracy import macro_accuracy
from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import benchmark_generate_config, benchmark_metrics
from dna_eval.tasks.benchmarks.kormedmcqa import STRIP, build_prompt, choice_patterns

REPO = "HAERAE-HUB/KMMLU-HARD"
# Commit this task is pinned to.
DATASET_REVISION = "58564a93d9f8d302a5a5ae41ab2f901dc303d5e1"
SUBJECTS = (
    "accounting", "agricultural_sciences", "aviation_engineering_and_maintenance", "biology",
    "chemical_engineering", "chemistry", "civil_engineering", "computer_science", "construction",
    "criminal_law", "ecology", "economics", "education", "electrical_engineering",
    "electronics_engineering", "energy_management", "environmental_science", "fashion",
    "food_processing", "gas_technology_and_engineering", "geomatics", "health", "industrial_engineer",
    "information_technology", "interior_architecture_and_design", "korean_history", "law",
    "machine_design_and_manufacturing", "management", "maritime_engineering", "marketing",
    "materials_engineering", "math", "mechanical_engineering", "nondestructive_testing", "patent",
    "political_science_and_sociology", "psychology", "public_safety", "railway_and_automotive_engineering",
    "real_estate", "refrigerating_machinery", "social_welfare", "taxation",
    "telecommunications_and_wireless_technology",
)
# File of each split of a subject, as the dataset card's ``configs`` maps them.
DATA_FILES = {"dev": "data/{}-dev.csv", "test": "data/{}-hard-test.csv"}
LETTERS = "ABCD"
NUM_FEWSHOT = 5


def read_rows(subject: str, split: str, revision: str) -> list[dict]:
    """The rows of a subject's ``split`` file at ``revision``, every value a string."""
    path = hf_hub_download(REPO, DATA_FILES[split].format(subject), repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_samples(subjects: list[str], num_fewshot: int, revision: str) -> list[Sample]:
    """Every test question of ``subjects``, each with the subject's first ``num_fewshot`` ``dev``
    examples; ids are ``<subject>-<row number>``. The CSV files are read directly at ``revision``
    instead of through ``datasets.load_dataset``, which is much slower for 90 small files."""
    unknown = sorted(set(subjects) - set(SUBJECTS))
    if unknown:
        raise ValueError(f"unknown KMMLU-HARD subjects {unknown}; choose from {list(SUBJECTS)}")
    samples: list[Sample] = []
    for subject in subjects:
        shots = read_rows(subject, "dev", revision)[:num_fewshot]
        for i, r in enumerate(read_rows(subject, "test", revision)):
            samples.append(Sample(id=f"{subject}-{i}", input=build_prompt(r, shots, LETTERS),
                                  target=LETTERS[int(r["answer"]) - 1], metadata={"subject": subject}))
    return samples


@task
def kmmlu_hard(thinking: bool = True, epochs: int = 1, subjects: list[str] | None = None,
               num_fewshot: int = NUM_FEWSHOT, revision: str = DATASET_REVISION) -> Task:
    """KMMLU-HARD, 4,104 test questions in 45 subjects x ``epochs`` runs; the headline is
    ``macro_accuracy``, the mean of the 45 subject accuracies, which is how the paper averages
    (lm-evaluation-harness reports the micro average, here ``accuracy``). ``subjects`` limits the
    run to some of them.

    Follows the paper's Direct evaluation (Son et al., 2024, arXiv:2402.11548, Table 4 and
    Appendix G; lm-evaluation-harness ``kmmlu_hard_direct``): five-shot, the examples taken from
    the ``dev`` split of the same subject, each question with its four options and a ``정답：``
    line.

    Deviations:

    - the answer is read with the KorMedMCQA answer-extraction cascade as ``kormedmcqa.py`` uses
      it (full-width colon accepted, bold markers removed): the reference compares the whole
      reply, cut at the first ``.`` or blank line, with the letter, which a reasoning model's
      final answer seldom is;
    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the reference decodes greedily with stop strings);
    - the prompt is one user turn and the answer is read from the final content only;
    - a truncated reply scores 0 even if it contains an answer.
    """
    return Task(
        dataset=MemoryDataset(load_samples(list(subjects or SUBJECTS), num_fewshot, revision), name="kmmlu_hard"),
        solver=generate(),
        scorer=letter_choice(choice_patterns(LETTERS), letters=LETTERS, strip=STRIP, uppercase=True),
        metrics=[macro_accuracy("subject"), *benchmark_metrics("subject")],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HeadlineMetric(metric="macro_accuracy"),
    )
