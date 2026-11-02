"""KMMLU-Pro: 2,822 multiple-choice questions from Korean professional licensing exams."""
from __future__ import annotations

import json

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import HeadlineMetric
from inspect_ai.solver import generate

from dna_eval.metrics.license_pass import license_pass
from dna_eval.metrics.weighted_accuracy import weighted_accuracy
from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import benchmark_generate_config, benchmark_metrics

REPO = "LGAI-EXAONE/KMMLU-Pro"       # gated: needs an access grant and a Hub token
DATA_FILE = "data/kmmlu_pro.jsonl"
# Commit this task is pinned to.
DATASET_REVISION = "e1567939724a369e428e66f15472562c55a5d181"

# LG-AI-EXAONE/KMMLU-Pro (BSD-3-Clause, Copyright (c) 2025 LG AI Research); see THIRD_PARTY_NOTICES.md.
# The official prompts. The second line holds a single space, as in the official code; the
# five-option variant adds the letter E to the instruction and an option line.
TEMPLATE = ("다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 {letters} 중 하나이고, \"정답:\" 뒤에 와야 합니다. "
            "정답을 고르기 전에 차근차근 생각하고 추론하세요.\n \n{question}\n\n{options}")
ANSWER_PATTERN = r"(?i)정답[^A-E]*:[^A-E]*([A-E])"
LETTERS = "ABCDE"


def build_prompt(question: str, options: list[str]) -> str:
    """The official Korean prompt for a question with four or five options."""
    if len(options) not in (4, 5):
        raise ValueError(f"KMMLU-Pro questions have 4 or 5 options, got {len(options)}")
    letters = LETTERS[:len(options)]
    lines = "\n".join(f"{letter}) {option}" for letter, option in zip(letters, options))
    return TEMPLATE.format(letters=letters, question=question, options=lines)


def record_to_sample(record: dict, index: int) -> Sample:
    """One question: ``solution`` is the 1-based number of the correct option; ``score`` is the
    exam points the question is worth, used as its weight."""
    return Sample(
        id=index,
        input=build_prompt(record["question"], record["options"]),
        target=LETTERS[int(record["solution"]) - 1],
        metadata={"license_name": record["license_name"], "subject": record["subject"],
                  "score": float(record["score"])},
    )


def load_samples(revision: str) -> list[Sample]:
    """The benchmark file at ``revision``; sample ids are row numbers."""
    path = hf_hub_download(REPO, DATA_FILE, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [record_to_sample(r, i) for i, r in enumerate(rows)]


@task
def kmmlu_pro(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """KMMLU-Pro, 2,822 questions x ``epochs`` runs; headline is the score-weighted accuracy.

    Follows LG-AI-EXAONE/KMMLU-Pro (``generate_model_responses.py``, ``utils.py``,
    ``print_score.py``): the Korean prompt with four or five options, one user turn, the
    ``정답:`` answer pattern, and the metrics. ``weighted_accuracy`` is the official headline
    (each question weighted by the exam points it is worth); ``accuracy`` is the plain
    mean; ``license_pass`` reports the weighted accuracy per licence and whether the licence
    would be passed under the official exam thresholds.

    The dataset is gated on the Hub: request access and provide a token (``HF_TOKEN``).

    Deviations, all for reasoning models:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the official script uses temperature 0 and 32,768 tokens);
    - the answer is read from the final content only (the official script reads the whole
      reply) and the last match is used, as the official script does;
    - a truncated reply scores 0 even if it contains an answer line;
    - the question file is read directly from ``data/kmmlu_pro.jsonl`` at the pinned commit
      instead of through ``datasets.load_dataset``.
    """
    return Task(
        dataset=MemoryDataset(load_samples(revision), name="kmmlu_pro"),
        solver=generate(),
        scorer=letter_choice([ANSWER_PATTERN], letters=LETTERS),
        metrics=[weighted_accuracy(), *benchmark_metrics(), license_pass()],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HeadlineMetric(metric="weighted_accuracy"),
    )
