"""AIME 2025: the 30 problems of AIME I and II 2025, with integer answers 0-999.

Source: MathArena (https://github.com/eth-sri/matharena): ``configs/competitions/aime/aime_2025.yaml``
for the prompt and ``src/matharena/parser.py`` for answer extraction; data ``MathArena/aime_2025``.

Followed from MathArena: the prompt (its instruction, then the problem, as one user turn); the
answer is the content of the last ``\\boxed{}`` / ``\\fbox{}`` (nested braces handled), or, with
no box, the last integer of the response (its ``strict_parsing: false`` setting for AIME); the
answer and the reference are compared as integers.

Departures:

- MathArena asks the model once more when a response holds no answer ("last chance" prompt);
  here that is a single turn, so a response without an answer is a format failure and scores 0.
- MathArena's parser, when a response has several boxes, prefers an earlier exact box over a
  last box that is only a decimal approximation; here the last box is always the answer.
- Thinking on, sampling from the served model's ``generation_config``, no ``max_tokens``, and
  the answer read from the final content only (decision record 0016). A truncated response
  scores 0 and is counted by ``answer_diagnostics``.
- ``epochs`` defaults to 8: accuracy is pass@1 averaged over 8 runs (MathArena's default is 4).
"""
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate

from dna_eval.scorers.math_answer import math_answer
from dna_eval.tasks.benchmarks import benchmark_generate_config

# MathArena/aime_2025 commit this task is pinned to.
DATASET_REVISION = "c94da77eb22bbd6439e62a323bec18493a421302"

# MathArena's ``instruction`` for AIME (the doubled braces in its YAML are ``str.format`` escapes).
INSTRUCTION = "Put your final answer within \\boxed{}.\nThe answer is an integer between 0 and 999 inclusive."


def record_to_sample(record: dict) -> Sample:
    """One problem as a single user turn: the instruction, a blank line, the problem."""
    return Sample(
        id=record["problem_idx"],
        input=f"{INSTRUCTION}\n\n{record['problem']}",
        target=str(record["answer"]),
    )


@task
def aime2025(thinking: bool = True, epochs: int = 8, revision: str = DATASET_REVISION) -> Task:
    """30 problems x ``epochs`` runs; the headline is ``accuracy`` (pass@1 averaged over runs)."""
    return Task(
        dataset=hf_dataset("MathArena/aime_2025", split="train", revision=revision, sample_fields=record_to_sample),
        solver=generate(),
        scorer=math_answer("integer"),
        epochs=epochs,
        config=benchmark_generate_config(thinking),
    )
