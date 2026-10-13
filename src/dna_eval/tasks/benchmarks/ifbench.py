"""IFBench: instruction following with constraints the model has not been trained on (``allenai/IFBench_test``)"""
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import HeadlineMetric
from inspect_ai.solver import generate

from dna_eval.scorers.instruction_following import instruction_following
from dna_eval.tasks.benchmarks import benchmark_generate_config

# allenai/IFBench_test commit this task is pinned to.
DATASET_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"

# The benchmark's README: "In the paper we generally report the prompt-level loose accuracy".
HEADLINE = HeadlineMetric(metric="prompt_level_loose")


def record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["key"],
        input=record["prompt"],
        target="",
        metadata={"instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"]},
    )


@task
def ifbench(thinking: bool = True, epochs: int = 8, revision: str = DATASET_REVISION) -> Task:
    """IFBench (Pyatkin et al., 2025): 300 single-turn prompts, each with one or more
    program-checkable constraints of 58 kinds (word and sentence patterns, formats, counts,
    ratios, ...) chosen to be unlike the instructions models were tuned on.

    Source followed: the benchmark's released code (``allenai/IFBench``: ``evaluation_lib.py``,
    ``run_eval.py``) and data (``allenai/IFBench_test`` on the Hub, pinned by ``revision``).
    The prompt is sent as it is, as one user turn. The final answer is checked by the benchmark's verifiers,
    strictly and loosely, and four numbers are reported: prompt-level and instruction-level
    accuracy, each strict and loose (``metrics/instruction_following.py``). The headline is
    prompt-level loose, the number the benchmark's paper and README lead with. How the two
    checks work, and the NLTK data the verifiers download on first use, are described in
    ``scorers/instruction_following.py``.

    Deviations, for reasoning models (decision record 0016):

    - the model is sampled with the served model's ``generation_config`` instead of greedy
      decoding, with thinking on and no output cap; only the final content is checked, never
      the reasoning;
    - a truncated response follows no instruction and is counted in ``truncated``;
    - sampling makes a single run noisy, so the task repeats the run ``epochs`` times (8 by
      default) and the metrics pool every (prompt, run) pair.
    """
    return Task(
        dataset=hf_dataset("allenai/IFBench_test", split="train", revision=revision, sample_fields=record_to_sample),
        solver=generate(),
        scorer=instruction_following(),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
