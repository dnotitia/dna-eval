"""KorMedMCQA: 3,009 questions from Korean doctor, nurse, pharmacist and dentist licensing exams."""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.solver import generate

from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics

REPO = "sean0042/KorMedMCQA"
# Commit this task is pinned to.
DATASET_REVISION = "79efd6f91edfc8036330d7a4daa88b9f2deb9a82"
EXAMS = ("doctor", "nurse", "pharm", "dentist")
LETTERS = "ABCDE"
NUM_FEWSHOT = 5


# lm-evaluation-harness (MIT, Copyright (c) 2020 EleutherAI), lm_eval/tasks/kormedmcqa; see
# THIRD_PARTY_NOTICES.md. The prompt of _template_yaml and the answer-extraction cascade of
# utils.py, which implements Appendix B of the KorMedMCQA paper.
def choice_patterns(letters: str) -> list[str]:
    """The extraction cascade for answers among ``letters``: tried in order, the last match of the
    first pattern that matches is the answer (case-insensitive, read upper-cased). The first
    pattern also accepts the full-width colon ``：`` the prompt itself uses (see the task)."""
    c = f"[{letters}]"
    return [rf"(?i)정답[:：\s]*({c})", rf"(?i)정답은\s*({c})\s*입니다", rf"(?i)\b({c})\.", rf"(?i)\b({c})\b"]


# Markdown bold markers, removed before the answer is read: ``**정답： A**``.
STRIP = ("**",)


def format_question(record: dict, letters: str) -> str:
    """A question and its options, ending with the ``정답：`` line (full-width colon)."""
    options = "\n".join(f"{letter}. {record[letter]}" for letter in letters)
    return f"{record['question'].strip()}\n{options}\n정답："


def build_prompt(record: dict, shots: list[dict], letters: str) -> str:
    """The few-shot prompt: each example with its answer letter, then the question."""
    examples = [f"{format_question(s, letters)} {letters[int(s['answer']) - 1]}" for s in shots]
    return "\n\n".join([*examples, format_question(record, letters)])


def load_samples(exams: list[str], num_fewshot: int, revision: str) -> list[Sample]:
    """Every test question of ``exams``, each with the exam's first ``num_fewshot`` examples of
    the ``fewshot`` split; ids are ``<exam>-<row number>``."""
    unknown = sorted(set(exams) - set(EXAMS))
    if unknown:
        raise ValueError(f"unknown KorMedMCQA exams {unknown}; choose from {list(EXAMS)}")
    raw = lambda r: Sample(input="", metadata=dict(r))  # noqa: E731
    samples: list[Sample] = []
    for exam in exams:
        shots = [s.metadata for s in hf_dataset(REPO, name=exam, split="fewshot", revision=revision,
                                                 sample_fields=raw)][:num_fewshot]
        for i, s in enumerate(hf_dataset(REPO, name=exam, split="test", revision=revision, sample_fields=raw)):
            r = s.metadata
            samples.append(Sample(id=f"{exam}-{i}", input=build_prompt(r, shots, LETTERS),
                                  target=LETTERS[int(r["answer"]) - 1],
                                  metadata={"exam": exam, "year": r["year"]}))
    return samples


@task
def kormedmcqa(thinking: bool = True, epochs: int = 1, exams: list[str] | None = None,
               num_fewshot: int = NUM_FEWSHOT, revision: str = DATASET_REVISION) -> Task:
    """KorMedMCQA, 3,009 test questions (doctor 435, nurse 878, pharmacist 885, dentist 811) x
    ``epochs`` runs; accuracy over all questions (the paper's and lm-evaluation-harness's group
    score, weighted by size) and by exam. ``exams`` limits the run to some of them.

    Follows the paper (Kweon et al., 2024, arXiv:2403.01469) and its lm-evaluation-harness task
    (``lm_eval/tasks/kormedmcqa``): five-shot, the examples taken from the dataset's ``fewshot``
    split of the same exam, each question with its five options and a ``정답：`` line, and the
    paper's answer-extraction cascade (Appendix B).

    Deviations:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the paper: temperature 0; lm-evaluation-harness: greedy, 1,024 tokens and stop strings);
    - the prompt is one user turn and the answer is read from the final content only;
    - the first extraction pattern also accepts the full-width colon (``정답： A``) and bold markers
      are removed first: a reasoning model's final answer often copies the prompt's ``정답：`` line
      in bold and then explains, and the reference cascade would then fall through to its last
      pattern and read a letter from the explanation;
    - a truncated reply scores 0 even if it contains an answer.
    """
    return Task(
        dataset=MemoryDataset(load_samples(list(exams or EXAMS), num_fewshot, revision), name="kormedmcqa"),
        solver=generate(),
        scorer=letter_choice(choice_patterns(LETTERS), letters=LETTERS, strip=STRIP, uppercase=True),
        metrics=benchmark_metrics("exam"),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
