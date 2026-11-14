"""KBL: Korean legal knowledge, legal reasoning and bar exam questions, in 65 tasks."""
from __future__ import annotations

import json
import re
from collections import defaultdict

from huggingface_hub import hf_hub_download
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (Metric, SampleScore, Score, Scorer, Target, Value, accuracy, grouped, metric,
                               scorer, stderr)
from inspect_ai.solver import TaskState, generate

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.scorers.letter_choice import EXPECTED
from dna_eval.tasks.benchmarks import HEADLINE, benchmark_generate_config
from dna_eval.utils.model_output import is_truncated
from dna_eval.utils.templates import render

REPO = "lbox/kbl"
# Commit this task is pinned to.
DATASET_REVISION = "695653ba87affa2c3113896013179c81fc16786a"

# lm-evaluation-harness (MIT, Copyright (c) 2020 EleutherAI), lm_eval/tasks/kbl: the task set, the
# description and the doc_to_text template of each task, as the YAML files parse; see
# THIRD_PARTY_NOTICES.md. DESCRIPTION ends in a backslash and an "n", not a newline: the YAML
# string is single-quoted.
DESCRIPTION = '당신은 사용자의 질문에 친절하고 논리적으로 답변해 주는 법률 전문가 챗봇 입니다.\\n'

# doc_to_text of each task family, as the YAML files define them (Jinja).
TEMPLATES = {
    'bar_exam': "### 질문: {{question}}\n다음 각 선택지를 읽고 A, B, C, D, E 중 하나를 선택하여 '답변: A' 와 같이 단답식으로 답해 주세요.\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nE. {{E}}\n### 답변:",
    'bar_exam_responsibility': "### 질문: {{question}}\n다음 각 선택지를 읽고 A, B, C, D 중 하나를 선택하여 '답변: A' 와 같이 단답식으로 답해 주세요.\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n### 답변:",
    'common_legal_mistake_qa': "### 질문: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\n'A', 'B', 'C' 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.",
    'legal_concept_qa': "### 질문: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nE. {{E}}\n'A', 'B', 'C', 'D', 'E' 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.",
    'offense_component_qa': "### 질문: {{question}}\n다음 각 선택지를 읽고 선택지 중 하나를 골라 ''답변: A'' 와 같이 단답식으로 답해 주세요. ### 선택지: A. {{A}}\nB. {{B}}.",
    'query_and_statute_matching_qa': "### 질문: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nE. {{E}}\nA, B, C, D, E 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.",
    'statute_hallucination_qa': "### 질문: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n'A', 'B', 'C', 'D' 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.",
    'statute_number_and_content_matching_qa': "### 질문: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nE. {{E}}\n A, B, C, D, E 중 하나를 선택하여 ''답변: A'' 와 같이 단답식으로 답해 주세요.",
    'case_relevance_qa_p': "### 질문: {{question}}\n\n[첫번째 판결문 상고인]\n{{query_case_appellant}}\n[첫번째 판결문 사실관계]\n{{query_case_fact}}\n[첫번째 판결문 당사자들의 주장]\n{{query_case_claim}}\n[첫번째 판결문 판사의 의견]\n{{query_case_judicial_opinion}}\n\n[두번째 판결문 상고인]\n{{retrieved_case_appellant}}\n[두번째 판결문 사실관계]\n{{retrieved_case_fact}}\n[두번째 판결문 당사자들의 주장]\n{{retrieved_case_claim}}\n[두번째 판결문 판사의 의견]\n{{retrieved_case_judicial_opinion}}\n\nA: {{A}}, B: {{B}}\n중 하나를 선택하여 '답변: A'과 같이 단답식으로 답해주세요.",
    'case_relevance_qa_q': "### 질문: {{question}}\n[의뢰인의 주장]\n{{query}}\n\n[판결문]\n- 상고인\n{{retrieved_case_appellant}}\n- 사실관계\n{{retrieved_case_fact}}\n- 당사자들의 주장\n{{retrieved_case_claim}}\n- 판사의 의견\n{{retrieved_case_judicial_opinion}}\n\nA: {{A}}, B: {{B}}\n중 하나를 선택하여 '답변: A'과 같이 단답식으로 답해주세요.",
    'causal_reasoning_qa': "### 질문: {{question}}\n검사의 공소사실: {{facts_charged}}\n피고인의 주장: {{defendant_claim}}\n증거: {{facts_accepted}}\nX, Y를 각각\nX: {{cause}})\nY: {{effect}}\n라고 할 때 X와 Y 사이의 관계를\nA: {{A}}, B: {{B}}\n중 하나를 선택하여 '답변: A'과 같이 단답식으로 답해주세요.",
    'statement_consistency_qa': "### 질문: {{question}}\n진술1: {{statement1}}\n진술2: {{statement2}}\nA: {{A}}, B: {{B}}\n중 하나를 선택하여 '답변: A'과 같이 단답식으로 답해주세요.",
}

# (dataset config, template, answer field, letters the answer pattern accepts) of every task.
TASKS = (
    ('bar_exam_civil_2012', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2013', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2014', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2015', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2016', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2017', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2018', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2019', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2020', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2021', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2022', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2023', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2024', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_civil_2025', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2012', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2013', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2014', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2015', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2016', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2017', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2018', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2019', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2020', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2021', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2022', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2023', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_criminal_2024', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2012', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2013', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2014', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2015', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2016', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2017', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2018', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2019', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2020', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2021', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2022', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2023', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_public_2024', 'bar_exam', 'gt', 'ABCDE'),
    ('bar_exam_responsibility_2010', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2011', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2012', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2013', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2014', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2015', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2016', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2017', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2018', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2019', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2020', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2021', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2022', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('bar_exam_responsibility_2023', 'bar_exam_responsibility', 'gt', 'ABCD'),
    ('kbl_knowledge_common_legal_mistake_qa', 'common_legal_mistake_qa', 'label', 'ABCDE'),
    ('kbl_knowledge_common_legal_mistake_qa_reasoning', 'common_legal_mistake_qa', 'label', 'ABCDE'),
    ('kbl_knowledge_legal_concept_qa', 'legal_concept_qa', 'label', 'ABCDE'),
    ('kbl_knowledge_offense_component_qa', 'offense_component_qa', 'label', 'ABCDE'),
    ('kbl_knowledge_query_and_statute_matching_qa', 'query_and_statute_matching_qa', 'label', 'ABCDE'),
    ('kbl_knowledge_statute_hallucination_qa', 'statute_hallucination_qa', 'label', 'ABCDE'),
    ('kbl_knowledge_statute_number_and_content_matching_qa', 'statute_number_and_content_matching_qa', 'label', 'ABCDE'),
    ('kbl_reasoning_case_relevance_qa_p', 'case_relevance_qa_p', 'label', 'ABCDE'),
    ('kbl_reasoning_case_relevance_qa_q', 'case_relevance_qa_q', 'label', 'ABCDE'),
    ('kbl_reasoning_causal_reasoning_qa', 'causal_reasoning_qa', 'label', 'ABCDE'),
    ('kbl_reasoning_statement_consistency_qa', 'statement_consistency_qa', 'label', 'ABCDE'),
)

# The file of each task other than the bar exams, as the dataset card's ``configs`` maps them; a
# bar exam ``bar_exam_<subject>_<year>`` is ``bar_exam/<subject>/<subject><year>.json``.
DATA_FILES = {
    'kbl_knowledge_common_legal_mistake_qa': 'knowledge/kbl_legal_common_legal_mistake_qa_v0.1.json',
    'kbl_knowledge_common_legal_mistake_qa_reasoning': 'knowledge/kbl_legal_common_legal_mistake_qa_reasoning_v0.1.json',
    'kbl_knowledge_legal_concept_qa': 'knowledge/kbl_legal_concept_qa_v0.1.json',
    'kbl_knowledge_offense_component_qa': 'knowledge/kbl_offense_component_qa_v0.1.json',
    'kbl_knowledge_query_and_statute_matching_qa': 'knowledge/kbl_query_and_statute_matching_qa_v0.1.json',
    'kbl_knowledge_statute_hallucination_qa': 'knowledge/kbl_statute_hallucination_qa_v0.1.json',
    'kbl_knowledge_statute_number_and_content_matching_qa':
        'knowledge/kbl_statute_matching_qa_number_and_content_matching_qa_v0.1.json',
    'kbl_reasoning_case_relevance_qa_p': 'reasoning/kbl_case_relevance_qa_p_v0.1.json',
    'kbl_reasoning_case_relevance_qa_q': 'reasoning/kbl_case_relevance_qa_q_v0.1.json',
    'kbl_reasoning_causal_reasoning_qa': 'reasoning/kbl_causal_reasoning_qa_v0.1.json',
    'kbl_reasoning_statement_consistency_qa': 'reasoning/kbl_logical_contradiction_qa_v0.1.json',
}


def data_file(config: str) -> str:
    if config.startswith("bar_exam_"):
        subject, year = config.removeprefix("bar_exam_").rsplit("_", 1)
        return f"bar_exam/{subject}/{subject}{year}.json"
    return DATA_FILES[config]


FAMILIES = ("knowledge", "reasoning", "bar_exam")
# The paper's averages (Table 2): the seven knowledge tasks; causal reasoning and statement
# consistency; the 2024 bar exams in civil, criminal and public law.
PAPER_AVERAGES = {
    "knowledge_avg": [t for t, *_ in TASKS if t.startswith("kbl_knowledge_")],
    "reasoning_avg": ["kbl_reasoning_causal_reasoning_qa", "kbl_reasoning_statement_consistency_qa"],
    "bar_exam_2024_avg": ["bar_exam_civil_2024", "bar_exam_criminal_2024", "bar_exam_public_2024"],
}


def family(config: str) -> str:
    return "bar_exam" if config.startswith("bar_exam_") else config.split("_")[1]


def record_to_sample(record: dict, config: str, template: str, answer_field: str, letters: str,
                     index: int) -> Sample:
    """One question: the description followed directly by the task's prompt, as one user turn."""
    return Sample(id=f"{config}-{index}", input=DESCRIPTION + render(TEMPLATES[template], **record),
                  target=str(record[answer_field]).strip(),
                  metadata={"task": config, "family": family(config), "letters": letters})


def read_rows(config: str, revision: str) -> list[dict]:
    """The questions of a task: its JSON file (a list of records) at ``revision``."""
    path = hf_hub_download(REPO, data_file(config), repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_samples(families: list[str], revision: str) -> list[Sample]:
    """Every test question of the tasks in ``families``; ids are ``<config>-<row number>``. The
    JSON files are read directly at ``revision`` instead of through ``datasets.load_dataset``,
    which is much slower for 65 small files."""
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown KBL families {unknown}; choose from {list(FAMILIES)}")
    samples: list[Sample] = []
    for config, template, answer_field, letters in TASKS:
        if family(config) in families:
            samples += [record_to_sample(r, config, template, answer_field, letters, i)
                        for i, r in enumerate(read_rows(config, revision))]
    return samples


def first_letter(text: str, letters: str) -> str | None:
    """The first of ``letters`` (capital) anywhere in ``text``: the reference's ``([A-E]).*``
    pattern (``[A-D]`` for the responsibility exams) with ``take_first``."""
    m = re.search(f"([{letters}])", text)
    return m.group(1) if m else None


@scorer(metrics=[accuracy(), stderr(), answer_diagnostics()])
def kbl_answer() -> Scorer:
    """Score 1.0 if the first answer letter in the final content is the target. A truncated reply
    scores 0. The one question whose answer is ``정답없음`` (no correct option) always scores 0,
    as in the reference."""

    async def score(state: TaskState, target: Target) -> Score:
        extracted = first_letter(state.output.completion, state.metadata["letters"])
        truncated = is_truncated(state.output)
        correct = extracted is not None and extracted == target.text and not truncated
        return Score(value=1.0 if correct else 0.0, answer=extracted,
                     metadata={TRUNCATED: truncated, EXTRACTED: extracted, EXPECTED: target.text})

    return score


@metric
def kbl_averages() -> Metric:
    """The paper's averages: the plain mean of the task accuracies in each ``PAPER_AVERAGES`` set
    (a set with no task in the run is left out)."""

    def compute(scores: list[SampleScore]) -> Value:
        by_task: dict[str, list[float]] = defaultdict(list)
        for s in scores:
            by_task[(s.sample_metadata or {})["task"]].append(s.score.as_float())
        out = {}
        for name, tasks in PAPER_AVERAGES.items():
            accs = [sum(by_task[t]) / len(by_task[t]) for t in tasks if by_task.get(t)]
            if accs:
                out[name] = sum(accs) / len(accs)
        return out

    return compute


@task
def kbl(thinking: bool = True, epochs: int = 1, families: list[str] | None = None,
        revision: str = DATASET_REVISION) -> Task:
    """KBL, the 65 tasks of its lm-evaluation-harness implementation (3,376 questions) x
    ``epochs`` runs: 7 legal knowledge tasks (510), 4 legal reasoning tasks (288) and 54 bar exams
    (2,578: civil 2012-2025, criminal and public 2012-2024, responsibility 2010-2023).
    ``families`` limits the run to ``knowledge``, ``reasoning`` and/or ``bar_exam``.

    KBL (Kim et al., 2024, EMNLP Findings, arXiv:2410.08731) publishes its evaluation as
    lm-evaluation-harness tasks (``lm_eval/tasks/kbl``); this task follows them: zero-shot, the
    common description followed by each task's prompt, and the answer read as the first capital
    letter among the accepted ones (``A``-``E``, ``A``-``D`` for the responsibility exams).

    Reported: ``accuracy`` over all questions (the headline; the paper reports no overall
    number), ``kbl_averages`` (the paper's knowledge, reasoning and 2024 bar exam averages, each
    a plain mean of task accuracies), ``accuracy`` by family (``family/<name>``) and by task
    (``task/<config>``), and ``answer_diagnostics``.

    Deviations:

    - thinking is on, no ``max_tokens``, sampling from the served model's ``generation_config``;
    - the description and the prompt are one user turn, and the answer is read from the final
      content only;
    - a truncated reply scores 0 even if it contains an answer;
    - the closed-book setting only (the retrieval-augmented variants are not included).
    """
    return Task(
        dataset=MemoryDataset(load_samples(list(families or FAMILIES), revision), name="kbl"),
        solver=generate(),
        scorer=kbl_answer(),
        metrics=[accuracy(), stderr(), kbl_averages(), answer_diagnostics(),
                 grouped(accuracy(), "family", all=False, name_template="family/{group_name}"),
                 grouped(accuracy(), "task", all=False, name_template="task/{group_name}")],
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HEADLINE,
    )
