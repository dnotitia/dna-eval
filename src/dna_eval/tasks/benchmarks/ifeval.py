"""IFEval: can the model follow verifiable instructions? (``google/IFEval``, 541 prompts)"""
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import HeadlineMetric
from inspect_ai.solver import generate

from dna_eval.scorers.instruction_following import instruction_following
from dna_eval.tasks.benchmarks import benchmark_generate_config

# google/IFEval commit this task is pinned to.
DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"

# The number the original paper leads with.
HEADLINE = HeadlineMetric(metric="prompt_level_strict")


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        id=record["key"],
        input=record["prompt"],
        target="",
        metadata={"instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"]},
    )


@task
def ifeval(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """Instruction-Following Eval (Zhou et al., 2023): 541 prompts, each with one or more
    instructions a program can check (no commas, at least 300 words, a JSON answer, ...).

    Source followed: the benchmark's released code (``google-research/instruction_following_eval``)
    and data (``google/IFEval`` on the Hub, pinned by ``revision``). The prompt is sent as it
    is: no system message, no template. Each instruction is checked on the final answer by
    the benchmark's own verifiers, strictly and loosely, and four numbers are reported:
    prompt-level and instruction-level accuracy, each strict and loose
    (``metrics/instruction_following.py``). The headline is prompt-level strict. How the two
    checks work, and the NLTK data the verifiers download on first use, are described in
    ``scorers/instruction_following.py``.

    Deviations, for reasoning models (decision record 0015):

    - the model is sampled with the served model's ``generation_config`` instead of greedy
      decoding, with thinking on and no output cap, so a long reasoning trace does not cut the
      answer off; only the final content is checked, never the reasoning;
    - a truncated response follows no instruction and is counted in ``truncated``;
    - the verifiers are the vendored IFBench copy of the original ones, which is identical
      to the released code for IFEval's 25 instructions (``vendor/ifbench/NOTICE.md``).

    ``epochs`` repeats the run; the metrics then pool every (prompt, run) pair.
    """
    return Task(
        dataset=hf_dataset("google/IFEval", split="train", revision=revision, sample_fields=_record_to_sample),
        solver=generate(),
        scorer=instruction_following(),
        epochs=epochs,
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
