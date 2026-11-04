"""HAERAE-VISION: 653 real Korean visual questions, each answer judged against its own checklist."""
from __future__ import annotations

from typing import Literal

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import HeadlineMetric
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ContentImage, ContentText
from inspect_ai.scorer import grouped, mean, stderr
from inspect_ai.solver import generate

from dna_eval.metrics.answer_diagnostics import answer_diagnostics
from dna_eval.metrics.judge_failures import judge_failures
from dna_eval.scorers.checklist_grade import checklist_grade
from dna_eval.tasks.benchmarks import benchmark_generate_config
from dna_eval.utils.images import image_data_uri

REPO = "HAERAE-HUB/HAERAE-VISION"
DATA_FILES = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
# Commit this task is pinned to.
DATASET_REVISION = "3dbac42df1cca2b42a5782b5e3fe2298bfe235d1"

# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2026 HAERAE Team
# The default system prompt of github.com/HAE-RAE/HAERAE-VISION (main.py); see THIRD_PARTY_NOTICES.md.
SYSTEM = "You are a helpful math assistant. Answer precisely in Korean if the prompt is Korean."
# SPDX-SnippetEnd

QuestionType = Literal["original", "explicit"]


def record_to_sample(record: dict, question_type: QuestionType) -> Sample:
    """One question: the system prompt, then a user turn with the question text followed by
    its images, as main.py builds it."""
    question = record[f"question_{question_type}"]
    images = [ContentImage(image=image_data_uri(img["bytes"])) for img in record["images"]]
    return Sample(
        id=int(record["question_idx"]),
        input=[ChatMessageSystem(content=SYSTEM), ChatMessageUser(content=[ContentText(text=question), *images])],
        metadata={"question": question, "checklist": list(record["checklist"]),
                  "category": record["category"], "source": record["source"]},
    )


def load_records(revision: str) -> list[dict]:
    """The rows of the dataset's parquet files at ``revision``, read directly: the dataset
    card's recorded split size (165) is out of date (653 rows), which ``datasets`` rejects."""
    rows: list[dict] = []
    for name in DATA_FILES:
        rows += pq.read_table(hf_hub_download(REPO, name, repo_type="dataset", revision=revision)).to_pylist()
    return rows


@task
def haerae_vision(question_type: QuestionType = "original", thinking: bool = True, epochs: int = 1,
                  judge_retries: int = 2, revision: str = DATASET_REVISION) -> Task:
    """HAERAE-VISION, 653 questions x ``epochs`` runs; needs a vision-language model under
    evaluation and a judge bound to the ``grader`` role.

    Each question comes in two forms: ``original``, the under-specified question as a user
    asked it, and ``explicit``, the same question with its context spelled out; both share the
    images and the checklist. ``question_type`` picks one (default ``original``, the form of
    the paper's headline results; Choi et al., 2026, arXiv:2601.06165).

    Follows github.com/HAE-RAE/HAERAE-VISION (``main.py``, ``score.py``, ``utils/``): the
    system prompt, the question text followed by its images, and the checklist judge
    (``scorers/checklist_grade.py``), whose K/N score is the question's score.

    Reported: ``mean`` (the headline) and ``stderr`` of the checklist score, ``mean`` per
    category (``category/<name>``), ``answer_diagnostics`` and ``judge_failures``.

    Deviations:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the reference: temperature 0.2, 512 tokens);
    - the images are sent as stored in the dataset (PNG); the reference re-encodes them as
      JPEG (quality 95) for API models;
    - the answer judged is the final content; a truncated or empty answer scores 0;
    - the checklist is shown to the judge as a Python list (the reference passes the CSV text
      of the same list);
    - a failed or unreadable judgment is retried and then scores 0 (the reference drops it);
    - the parquet files are read directly at the pinned commit (see ``load_records``).
    """
    samples = [record_to_sample(r, question_type) for r in load_records(revision)]
    return Task(
        dataset=MemoryDataset(samples, name=f"haerae_vision_{question_type}"),
        solver=generate(),
        scorer=checklist_grade(judge_retries=judge_retries),
        metrics=[mean(), stderr(), answer_diagnostics(), judge_failures(),
                 grouped(mean(), "category", all=False, name_template="category/{group_name}")],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HeadlineMetric(metric="mean"),
    )
