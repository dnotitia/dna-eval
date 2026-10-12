"""CLIcK: 1,995 multiple-choice questions on Korean culture and language."""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import HeadlineMetric
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_ai.solver import generate

from dna_eval.metrics.answer_diagnostics import answer_diagnostics
from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import benchmark_generate_config

# EunsuKim/CLIcK commit this task is pinned to.
DATASET_REVISION = "d61627859645b5e6edc03fd9f835735d8226fa4e"

ANSWER_PATTERN = r"(?i)정답[^A-E]*:[^A-E]*([A-E])"
LETTERS = "ABCDE"

WITH_PARAGRAPH = ("주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 {letters} 중에 골라 알파벳 하나로 답하시오. "
                  "최종 정답은 \"정답:\" 뒤에 알파벳 하나로 쓰시오.\n\n맥락: {paragraph}\n질문: {question}\n보기:\n{options}")
WITHOUT_PARAGRAPH = ("주어진 질문을 천천히 읽고, 적절한 정답을 {letters} 중에 골라 알파벳 하나로 답하시오. "
                     "최종 정답은 \"정답:\" 뒤에 알파벳 하나로 쓰시오.\n\n질문: {question}\n보기:\n{options}")


def subcategory(record: dict) -> str:
    """The question's subcategory, one of the benchmark's eleven (derived from its id and,
    for two language subcategories, its wording; every question falls in exactly one).

    Language: ``textual``, ``grammar``, ``functional``. Culture: ``economy``, ``geography``,
    ``history``, ``law``, ``politics``, ``popular``, ``society``, ``tradition``.
    """
    i, q = record["id"], record["question"]
    parts = i.split("_")
    dialogue = "대화" in q or "발화" in q or "질의" in q
    if "CSAT_korean" in i:
        if int(parts[-1]) > 34 or (int(parts[2]) < 21 and int(parts[3]) < 11):
            return "functional"
        if int(parts[2]) < 21 and int(parts[3]) > 10:
            return "grammar"
        return "textual"
    if "TK" in i:
        return "grammar" if int(parts[-1]) < 5 else "textual"
    if "PSE_korean" in i:
        return "functional"
    if "Kedu_1" in i:
        return "functional" if parts[1] == "16" and dialogue else "grammar"
    lowered = i.lower()
    for name, key in (("economy", "economy"), ("geography", "geography"), ("politics", "politics"),
                      ("popular", "popular"), ("society", "society"), ("tradition", "tradition")):
        if key in lowered:
            return name
    if "KHB" in i or "history" in lowered:
        return "history"
    if "law" in lowered or "PSAT" in i:
        return "law"
    raise ValueError(f"cannot place CLIcK question {i!r} in a subcategory")


def build_prompt(record: dict) -> str:
    """The question with its paragraph (if any) and all of its lettered options."""
    choices = record["choices"]
    letters = ", ".join(LETTERS[:len(choices)])
    options = ", ".join([f"A:{choices[0]}"] + [f"{LETTERS[i]}: {c}" for i, c in enumerate(choices) if i])
    template = WITH_PARAGRAPH if record["paragraph"] else WITHOUT_PARAGRAPH
    return template.format(letters=letters, paragraph=record["paragraph"], question=record["question"],
                           options=options)


def record_to_sample(record: dict) -> Sample:
    """One question; ``answer`` is the text of the correct option and is mapped to its letter.

    The data's ``id`` is not unique (1,987 distinct values in 1,995 rows), so it is kept as
    ``metadata["source_id"]`` and the loader numbers the samples.
    """
    sub = subcategory(record)
    return Sample(
        input=build_prompt(record),
        target=LETTERS[record["choices"].index(record["answer"])],
        metadata={"category": "Language" if sub in ("textual", "grammar", "functional") else "Culture",
                  "subcategory": sub, "source_id": record["id"]},
    )


@task
def click(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """CLIcK, 1,995 questions x ``epochs`` runs; accuracy overall, per category and per subcategory.

    CLIcK's authors released no chat protocol; the prompts follow the ``click`` task of
    lm-evaluation-harness (instruction, ``맥락:``/``질문:``/``보기:`` layout, with a paragraph
    or without), and the ``category`` (Culture, Language) and ``subcategory`` groups follow its
    subtask definitions. The answer is the option text in the data and is mapped to a letter.

    Deviations from lm-evaluation-harness, which scores log-likelihoods of the letters:

    - all options are listed: 256 questions have a fifth option that the harness drops from
      the prompt, and the letters named in the instruction follow the number of options;
    - the instruction adds that the final answer follows ``정답:``, and the trailing ``정답:``
      cue is removed. A model that reasons first needs a marked final line to read; the form
      is the one KMMLU-Pro uses, so the two Korean tasks read answers the same way;
    - the answer is the last match of the KMMLU-Pro pattern ``정답[^A-E]*:[^A-E]*([A-E])`` in
      the final content; thinking is on, no ``max_tokens``, sampling from the served model's
      ``generation_config``;
    - a truncated reply scores 0 even if it contains an answer line.
    """
    return Task(
        dataset=hf_dataset("EunsuKim/CLIcK", split="train", revision=revision, sample_fields=record_to_sample,
                           auto_id=True),
        solver=generate(),
        scorer=letter_choice([ANSWER_PATTERN], letters=LETTERS),
        metrics=[accuracy(), stderr(), answer_diagnostics(),
                 grouped(accuracy(), "category", all=False, name_template="category/{group_name}"),
                 grouped(accuracy(), "subcategory", all=False, name_template="subcategory/{group_name}")],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HeadlineMetric(metric="accuracy"),
    )
