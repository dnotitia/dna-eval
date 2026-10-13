"""Humanity's Last Exam (HLE), text-only subset, judged by the model bound to the ``grader`` role.

Official source (github.com/centerforaisafety/hle, ``hle_eval/``), followed as written:

- data: Hub dataset ``cais/hle`` (gated; needs an accepted licence and a Hub token), ``test``
  split, pinned to a commit;
- subject prompt (``run_model_predictions.py``): one system message with the answer format
  (Explanation / Answer / Confidence), then the question as the user message;
- judge (``run_judge_results.py``): the official judge prompt, which extracts the final answer,
  says whether it matches the reference and extracts the stated confidence;
- metrics: accuracy, and the RMS calibration error computed as ``dump_metrics`` does
  (``metrics/calibration_error.py``, including its last-bin quirk).

Deviations, all recorded in the log:

- **Text-only.** The 342 questions that come with an image are left out (2,158 of 2,500 run),
  because the models evaluated here read text only. Numbers are comparable to a "text-only"
  column, not to the full-set score. The official script sends the image to multimodal models.
- **Judge.** The official judge is o3-mini with structured outputs. Here it is whatever model
  the ``grader`` role is bound to (required). A different judge shifts accuracy and calibration;
  compare only runs with the same judge. Structured output is requested through Inspect's
  ``response_schema``; see ``scorers/hle_judge.py`` for the fallback and the retries.
- **Reasoning-model handling** (decision record 0016): thinking on, sampling from the served
  model's ``generation_config``, no ``max_tokens`` (the official script caps the subject by
  ``--max_completion_tokens``), the judge sees only the final content. A cut-off answer scores 0
  without a judge call and is reported by ``answer_diagnostics``.
- Single run (``epochs`` 1), as in the official protocol.
"""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import generate

from dna_eval.scorers.hle_judge import hle_judge
from dna_eval.metrics.calibration_error import calibration_error
from dna_eval.metrics.judge_failures import judge_failures
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics

HF_DATASET = "cais/hle"
# cais/hle commit this task is pinned to (decision record 0008).
DATASET_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"

# Verbatim from run_model_predictions.py: one system prompt for every question type.
# centerforaisafety/hle (MIT, Copyright (c) 2025 centerforaisafety); see THIRD_PARTY_NOTICES.md.
SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def load_samples(revision: str = DATASET_REVISION) -> list[Sample]:
    """The text-only questions of ``cais/hle`` at ``revision`` (rows whose ``image`` is empty).

    Only the columns used are read, so no image decoder is needed. The sample input is the
    official message pair; ``category`` and ``answer_type`` go to the sample metadata.
    """
    from datasets import load_dataset

    rows = load_dataset(HF_DATASET, split="test", revision=revision).select_columns(
        ["id", "question", "image", "answer", "answer_type", "category"])
    return [Sample(id=r["id"],
                   input=[ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=r["question"])],
                   target=r["answer"],
                   metadata={"category": r["category"], "answer_type": r["answer_type"]})
            for r in rows if not r["image"]]


@task
def hle(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """HLE text-only (2,158 questions), one run by default. Headline: ``accuracy``.

    Reports ``accuracy`` and its ``stderr``, accuracy by ``category``, the 0-100
    ``calibration_error`` and ``answer_diagnostics``. ``thinking`` is forwarded as
    ``chat_template_kwargs.enable_thinking``. Needs ``--model-role grader=<judge>``.
    """
    return Task(
        dataset=MemoryDataset(load_samples(revision), name="hle-text-only"),
        solver=generate(),
        scorer=hle_judge(),
        metrics=[*benchmark_metrics("category"), calibration_error(), judge_failures()],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
