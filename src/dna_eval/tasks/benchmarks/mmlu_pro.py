"""MMLU-Pro: 12,032 multiple-choice questions in 14 subjects, up to ten options each."""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate

from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics

# TIGER-Lab/MMLU-Pro commit this task is pinned to.
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

INSTRUCTION = ("The following are multiple choice questions (with answers) about {category}. Think step by "
               "step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")
# The official script's two extraction patterns, in order.
ANSWER_PATTERNS = [r"answer is \(?([A-J])\)?", r"[aA]nswer:\s*([A-J])"]
LETTERS = "ABCDEFGHIJ"


def record_to_sample(record: dict) -> Sample:
    """One question as the official script formats a test item, after its instruction line.

    Options equal to ``"N/A"`` (padding in questions with fewer than ten) are dropped.
    """
    options = [o for o in record["options"] if o != "N/A"]
    item = f"Question: {record['question']}\nOptions: " + "".join(f"{LETTERS[i]}. {o}\n" for i, o in enumerate(options))
    return Sample(
        id=record["question_id"],
        input=INSTRUCTION.format(category=record["category"]) + item + "Answer: Let's think step by step.\n\n",
        target=record["answer"],
        metadata={"category": record["category"], "src": record["src"], "n_options": len(options)},
    )


@task
def mmlu_pro(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """MMLU-Pro test split, 12,032 questions x ``epochs`` runs; micro accuracy plus accuracy per category.

    Follows TIGER-AI-Lab/MMLU-Pro (``evaluate_from_api.py``): the same instruction line, item
    layout (``Question:``, ``Options:`` with lettered lines, ``Answer: Let's think step by
    step.``), ``**`` removed from the reply, and the first two steps of the answer chain, ``answer
    is (X)`` then ``Answer: X``.

    Deviations:

    - 0-shot: the official script puts five chain-of-thought examples of the question's
      category in front; reasoning models are reported without them;
    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the official script uses temperature 0 and 4,000 tokens);
    - the *last* match of the first pattern that matches is used (the official one takes the
      first); the second pattern is the official ``.*[aA]nswer:\\s*([A-J])`` without its
      leading ``.*``, so that too reads the last occurrence;
    - the official chain's third step (any standalone capital letter) and its random-guess
      fallback are dropped: a reply without an answer line scores 0 and counts as a format
      failure;
    - a truncated reply scores 0 even if it contains an answer line.
    """
    return Task(
        dataset=hf_dataset("TIGER-Lab/MMLU-Pro", split="test", revision=revision, sample_fields=record_to_sample),
        solver=generate(),
        scorer=letter_choice(ANSWER_PATTERNS, letters=LETTERS, strip=("**",)),
        metrics=benchmark_metrics("category"),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
