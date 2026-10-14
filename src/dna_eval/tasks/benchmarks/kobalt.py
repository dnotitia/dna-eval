"""KoBALT-700: 700 expert-written Korean linguistics questions with ten options each."""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.solver import generate, system_message

from dna_eval.scorers.letter_choice import letter_choice
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config, benchmark_metrics

# snunlp/KoBALT-700 commit this task is pinned to.
DATASET_REVISION = "30c30a431066508e6bef77cfa6d6059b85b12f0d"

# SPDX-SnippetBegin
# SPDX-License-Identifier: CC-BY-NC-4.0
# SPDX-SnippetCopyrightText: 2025 Hyopil Shin, Sangah Lee, Dongjun Jang, Wooseok Song, Jaeyoon Kim,
#   Chaeyoung Oh, Hyemi Jo, Youngchae Ahn, Sihyun Oh, Hyohyeong Chang, Sunkyoung Kim, Jinsik Lee
# The KoBALT authors' prompts and answer pattern (github.com/snunlp/KoBALT). NOT under the Apache
# License: CC-BY-NC-4.0, non-commercial use only, like the KoBALT dataset. See THIRD_PARTY_NOTICES.md.
SYSTEM = "당신은 문제를 해결하는 전문가입니다."
# The question already contains its lettered options and usually ends with a newline.
PROMPT = ("다음 문제에 대해서 충분히 생각하고 추론하여, 10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.\n\n"
          "{question}\n\n"
          "답변은 반드시 다음 형식을 엄격히 지켜야 합니다: \"정답은 [정답 보기]입니다.\"로 끝나야 하고, "
          "[정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다.\n"
          "정답: 문제를 풀기 위해, 한 번 천천히 생각해봅시다.")
ANSWER_PATTERN = r"정답은\s*(.*?)\s*입니다"
# SPDX-SnippetEnd

LETTERS = "ABCDEFGHIJ"


def record_to_sample(record: dict) -> Sample:
    """One question; the options are part of ``Question``."""
    return Sample(
        id=record["ID"],
        input=PROMPT.format(question=record["Question"]),
        target=record["Answer"].strip(),
        metadata={"class": record["Class"], "subclass": record["Subclass"], "level": record["Level"]},
    )


@task
def kobalt(thinking: bool = True, epochs: int = 1, revision: str = DATASET_REVISION) -> Task:
    """KoBALT-700 (``kobalt_v1``, ``raw`` split), 700 questions x ``epochs`` runs; accuracy overall,
    by linguistic class and by difficulty level.

    Follows the benchmark's evaluation protocol (``evaluation_protocol.md`` and
    ``data/inference_gpt.py`` in github.com/snunlp/KoBALT): the Korean system prompt, the user prompt
    around the question, and the answer rule. A reply is read from every ``정답은 ... 입니다``
    statement; it is correct only if the statements together name exactly one distinct letter
    and it is the right one, so a reply that names several letters is wrong.

    The prompts and the answer pattern are the authors' text, marked above as a CC-BY-NC-4.0
    snippet (non-commercial use only); the dataset has the same licence.

    Deviations:

    - the user prompt is the protocol's JSON form, with a newline before the final ``정답:``
      line; the reference script runs the two sentences together, and spells ``끝나야하고``
      without a space;
    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``
      (the reference caps output at 2,048 tokens and decodes greedily);
    - only the letters A-J count as answers; the reference takes every capital letter
      inside a statement (``[A-Z]``);
    - a truncated reply scores 0 even if it contains an answer statement.
    """
    return Task(
        dataset=hf_dataset("snunlp/KoBALT-700", name="kobalt_v1", split="raw", revision=revision,
                           sample_fields=record_to_sample),
        solver=[system_message(SYSTEM), generate()],
        scorer=letter_choice([ANSWER_PATTERN], letters=LETTERS, unique_letter=True),
        metrics=benchmark_metrics("class", "level"),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
