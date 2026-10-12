"""HRM8K KSM: 1,428 Korean math problems (competition, university and teacher-exam level).

Source: the HRM8K authors publish the data (``HAERAE-HUB/HRM8K``, config ``KSM``, split ``test``)
but no evaluation code, so this follows the most used reference, lm-evaluation-harness
(``lm_eval/tasks/hrm8k/default``: ``utils.py`` and ``_hrm8k_yaml``).

Followed from lm-evaluation-harness: the Korean zero-shot prompt (verbatim, ending in ``답변:``);
the answer is the content of the last ``\\boxed{}`` with the text after its last ``=`` kept, and
for a response without a box the last ``$...$`` span or the last number; the answer and the
reference are compared as strings after the Hendrycks et al. MATH normalisation.

Departures:

- An answer also counts as correct when ``math-verify`` finds it mathematically equal to the
  reference. 222 reference answers are LaTeX expressions (``\\frac{1}{6}``, ``8\\pi``, sets,
  intervals) that string comparison judges wrong for an equal answer written another way.
- The reference is normalised directly. lm-evaluation-harness reads it with the response
  parser, which reduces a non-numeric reference such as ``8\\pi`` to its last number.
- Only the final content is read, and a response without a box is still read by the fallbacks
  above (as in lm-evaluation-harness), so ``answer_diagnostics``' format-failure rate counts
  only responses with no readable answer; ``boxed`` in the score metadata tells them apart.
- Thinking on, sampling from the served model's ``generation_config``, and no output cap
  (lm-evaluation-harness: greedy, 2048 tokens), which would cut off a reasoning model
  mid-thought (decision record 0015). A truncated response scores 0.

Reported: ``accuracy``, ``stderr``, ``accuracy`` per ``category`` (``grouped``), and
``answer_diagnostics``.
"""
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_ai.solver import generate

from dna_eval.metrics.answer_diagnostics import answer_diagnostics
from dna_eval.scorers.math_answer import math_answer
from dna_eval.tasks.benchmarks import benchmark_generate_config

# HAERAE-HUB/HRM8K commit this task is pinned to.
DATASET_REVISION = "c360cabf8d733a82455565358b3dc965aab9ba8d"


def record_to_sample(record: dict) -> Sample:
    """One problem as a single user turn, in the prompt of lm-evaluation-harness's HRM8K task."""
    prompt = ("주어진 문제를 풀어보세요.\n"
              "문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
              f"문제: {record['question'].strip()}\n답변:")
    return Sample(input=prompt, target=str(record["answer"]),
                  metadata={"category": record["category"], "difficulty": record["difficulty"]})


@task
def hrm8k_ksm(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """1,428 problems x ``epochs`` runs; the headline is ``accuracy``."""
    return Task(
        dataset=hf_dataset("HAERAE-HUB/HRM8K", name="KSM", split="test", revision=revision,
                           sample_fields=record_to_sample, auto_id=True),
        solver=generate(),
        scorer=math_answer("expression"),
        metrics=[accuracy(), stderr(), grouped(accuracy(), "category"), answer_diagnostics()],
        epochs=epochs,
        config=benchmark_generate_config(thinking),
    )
