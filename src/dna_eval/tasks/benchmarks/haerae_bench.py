"""HAE-RAE Bench 1.1: Korean knowledge and language questions in 11 multiple-choice subsets."""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.solver import generate

from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics
from dna_eval.tasks.benchmarks.kmmlu_pro import ANSWER_PATTERN

REPO = "HAERAE-HUB/HAE_RAE_BENCH_1.1"
# Commit this task is pinned to.
DATASET_REVISION = "b480e81024913f27783d2b05d2f0b10089db19ad"

# The multiple-choice subsets, each a dataset config with a single ``test`` split. The two
# generative subsets (``lyrics_denoising``, ``proverbs_denoising``) have no published scoring
# rule and are left out.
SUBSETS = (
    "correct_definition_matching", "csat_geo", "csat_law", "csat_socio", "date_understanding",
    "general_knowledge", "history", "loan_words", "rare_words", "reading_comprehension",
    "standard_nomenclature",
)
LETTERS = "ABCDE"

# The answer-format sentence of the KMMLU-Pro prompt (LG-AI-EXAONE/KMMLU-Pro, BSD-3-Clause; see
# THIRD_PARTY_NOTICES.md), with its letters, placed before the benchmark's own prompt.
INSTRUCTION = "다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 ABCDE 중 하나이고, \"정답:\" 뒤에 와야 합니다."


def build_prompt(query: str) -> str:
    """The answer-format instruction, then the subset's prompt as published (``query``, which
    already holds the question and its options (A)-(E))."""
    return f"{INSTRUCTION}\n\n{query}"


def record_to_sample(record: dict, subset: str, index: int) -> Sample:
    """One question; ``answer`` is the correct letter in parentheses, e.g. ``(B)``."""
    target = record["answer"].strip().strip("()")
    if target not in LETTERS or len(target) != 1:
        raise ValueError(f"{subset} row {index}: answer {record['answer']!r} is not one of (A)-(E)")
    return Sample(id=f"{subset}-{index}", input=build_prompt(record["query"]), target=target,
                  metadata={"subset": subset})


def load_samples(subsets: list[str], revision: str) -> list[Sample]:
    """Every question of ``subsets`` at ``revision``; ids are ``<subset>-<row number>``."""
    unknown = sorted(set(subsets) - set(SUBSETS))
    if unknown:
        raise ValueError(f"unknown HAE-RAE Bench subsets {unknown}; choose from {list(SUBSETS)}")
    samples: list[Sample] = []
    for subset in subsets:
        rows = hf_dataset(REPO, name=subset, split="test", revision=revision,
                          sample_fields=lambda r: Sample(input="", metadata=dict(r)))
        samples += [record_to_sample(s.metadata, subset, i) for i, s in enumerate(rows)]
    return samples


@task
def haerae_bench(thinking: bool = True, epochs: int = 1, subsets: list[str] | None = None,
                 revision: str = DATASET_REVISION) -> Task:
    """HAE-RAE Bench 1.1, the 11 multiple-choice subsets (3,606 questions) x ``epochs`` runs;
    accuracy over all questions and by subset. ``subsets`` limits the run to some of them.

    The benchmark's own prompt is used as published: each row's ``query`` holds the
    instruction, the question and the five options, and ends with ``### 정답:`` (the CSAT
    subsets end with the options). The original HAE-RAE Bench paper (Son et al., 2023,
    arXiv:2309.02706) picks the option with the highest log-likelihood, through
    lm-evaluation-harness; no evaluation code for version 1.1 is published.

    Deviations:

    - the model generates its answer instead of being scored by option log-likelihood, which a
      chat endpoint cannot provide; an answer-format sentence taken from the KMMLU-Pro prompt
      is placed before the query, and the answer is read with the KMMLU-Pro ``정답:`` pattern
      (last match, final content only);
    - the two generative subsets (``lyrics_denoising``, ``proverbs_denoising``) are left out:
      no scoring rule is published for them;
    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``;
    - a truncated reply scores 0 even if it contains an answer line.
    """
    return Task(
        dataset=MemoryDataset(load_samples(list(subsets or SUBSETS), revision), name="haerae_bench"),
        solver=generate(),
        scorer=letter_choice([ANSWER_PATTERN], letters=LETTERS),
        metrics=benchmark_metrics("subset"),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
