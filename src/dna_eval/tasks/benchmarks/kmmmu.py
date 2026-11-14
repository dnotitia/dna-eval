"""KMMMU: 3,466 Korean exam questions (civil service, qualification, NCS, olympiad) with images."""
from __future__ import annotations

import ast
import csv

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_ai.solver import generate

from dna_eval.metrics.answer_diagnostics import answer_diagnostics
from dna_eval.metrics.judge_failures import judge_failures
from dna_eval.scorers.answer_judge import answer_judge
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config

REPO = "HAERAE-HUB/KMMMU"
DATA_FILE = "kmmmu.csv"
# Commit this task is pinned to.
DATASET_REVISION = "becfb28fe19e66f5ce5bdbf61e9073b44609cc5e"

# github.com/HAE-RAE/KMMMU (kmmmu_utils.py), which states no licence; see THIRD_PARTY_NOTICES.md.
SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "You MUST output the final answer in boxed format as LaTeX with dollar signs, exactly like:\n"
    "$\\boxed{ANSWER}$\n"
    "Do NOT add any extra text outside the box."
)


def record_to_sample(record: dict, index: int) -> Sample:
    """One question: the system prompt, then a user turn with the question text followed by
    its images (``image_link``, a list written as a Python literal), as the tutorial builds it."""
    images = [ContentImage(image=url) for url in ast.literal_eval(record["image_link"])]
    return Sample(
        id=index,
        input=[ChatMessageSystem(content=SYSTEM_PROMPT),
               ChatMessageUser(content=[ContentText(text=record["question"]), *images])],
        target=record["answer"],
        metadata={"question": record["question"], "question_type": record["question_type"]},
    )


def load_samples(revision: str) -> list[Sample]:
    """``kmmmu.csv`` at ``revision``; sample ids are row numbers."""
    path = hf_hub_download(REPO, DATA_FILE, repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8", newline="") as f:
        return [record_to_sample(r, i) for i, r in enumerate(csv.DictReader(f))]


@task
def kmmmu(thinking: bool = True, epochs: int = 1, judge_retries: int = 2, revision: str = DATASET_REVISION) -> Task:
    """KMMMU, 3,466 questions x ``epochs`` runs; needs a vision-language model under evaluation,
    a judge bound to the ``grader`` role, and ``math-verify`` (``uv sync --extra math``).

    Follows the authors' evaluation tutorial (github.com/HAE-RAE/KMMMU, ``eval_tutorial.ipynb``
    and ``kmmmu_utils.py``; Lee et al., 2026, arXiv:2604.13058): the system prompt asking for
    a ``$\\boxed{}$`` answer, the question text followed by its images, a math-verify parse of
    the answer, and a judge that compares the answer with the gold answer and replies true or
    false (``scorers/answer_judge.py``).

    Reported: ``accuracy`` (the headline), ``accuracy`` per question type
    (``question_type/<type>``: single- and multiple-answer multiple choice, numeric, short text
    and essay), ``answer_diagnostics`` and ``judge_failures``.

    The images are fetched from their URLs when the task runs. About two thirds are hosted
    outside the Hub, on the exam sites the questions come from, so a run depends on those
    sites being reachable.

    Deviations:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the tutorial: temperature 0.6, 8,192 tokens);
    - the images are sent as fetched (JPEG, PNG or GIF); the tutorial decodes each one and
      re-encodes it as PNG;
    - the answer parsed and judged is the final content (the tutorial uses the whole generated
      text); a truncated or empty answer scores 0 without a judgment;
    - a failed judge call is retried, then scores 0.
    """
    return Task(
        dataset=MemoryDataset(load_samples(revision), name="kmmmu"),
        solver=generate(),
        scorer=answer_judge(judge_retries=judge_retries),
        metrics=[accuracy(), stderr(), answer_diagnostics(), judge_failures(),
                 grouped(accuracy(), "question_type", all=False, name_template="question_type/{group_name}")],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
