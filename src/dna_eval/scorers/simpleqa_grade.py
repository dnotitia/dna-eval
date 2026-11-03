"""``simpleqa_grade``: grade a short factual answer as SimpleQA does, with the ``grader`` model.

The grader reads the question, the gold target and the model's final content, and answers
A (CORRECT), B (INCORRECT) or C (NOT_ATTEMPTED). The grader prompt, its system message, its
temperature and the way the letter is read are those of OpenAI's simple-evals
(``simpleqa_eval.py``); ``simpleqa_metrics`` reports SimpleQA's aggregate numbers.
"""
from __future__ import annotations

import asyncio

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, accuracy, metric, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.answer_diagnostics import EXTRACTED, TRUNCATED, answer_diagnostics
from dna_eval.metrics.judge_failures import JUDGE_ERROR, judge_failures
from dna_eval.roles import GRADER_ROLE
from dna_eval.utils.model_output import is_truncated

# SPDX-SnippetBegin
# SPDX-License-Identifier: MIT
# SPDX-SnippetCopyrightText: 2024 OpenAI
# From github.com/openai/simple-evals (simpleqa_eval.py, sampler/chat_completion_sampler.py);
# see THIRD_PARTY_NOTICES.md. Three lines of the template end in a space, as published; keep them.
GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

GRADER_SYSTEM_MESSAGE = "You are a helpful assistant."
GRADER_TEMPERATURE = 0.5
# SPDX-SnippetEnd

GRADES = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
GRADE = "grade"                # Score.metadata key: one of GRADES' values


def read_grade(reply: str) -> str | None:
    """The first A, B or C in the grader's reply, as simple-evals reads it; ``None`` if there is none."""
    return next((c for c in reply if c in GRADES), None)


@metric(scores="unreduced")
def simpleqa_metrics() -> Metric:
    """SimpleQA's aggregate numbers over every graded answer (each run counted):
    ``correct``, ``incorrect``, ``not_attempted``, ``correct_given_attempted`` (correct among
    the attempted) and ``f_score`` (the harmonic mean of ``correct`` and
    ``correct_given_attempted``)."""

    def compute(scores: list[SampleScore]) -> Value:
        n = len(scores)
        grades = [(s.score.metadata or {}).get(GRADE) for s in scores]
        rate = {g: (sum(1 for x in grades if x == g) / n if n else 0.0) for g in GRADES.values()}
        attempted = rate["CORRECT"] + rate["INCORRECT"]
        cga = rate["CORRECT"] / attempted if attempted else 0.0
        f = 2 * cga * rate["CORRECT"] / (cga + rate["CORRECT"]) if cga + rate["CORRECT"] else 0.0
        return {"correct": rate["CORRECT"], "incorrect": rate["INCORRECT"],
                "not_attempted": rate["NOT_ATTEMPTED"], "correct_given_attempted": cga, "f_score": f}

    return compute


@scorer(metrics=[accuracy(), stderr(), simpleqa_metrics(), answer_diagnostics(), judge_failures()])
def simpleqa_grade(question_key: str = "question", judge_retries: int = 2) -> Scorer:
    """Score 1.0 for a CORRECT grade, else 0.0; ``metadata[GRADE]`` holds the grade.

    The question shown to the grader is ``state.metadata[question_key]``, the gold target is
    the sample's target, and the predicted answer is the final content (never the reasoning).

    - A truncated or empty answer is NOT_ATTEMPTED without asking the grader (KoSimpleQA marks
      a reasoning model that never finished its answer as not attempted).
    - A grader call that fails is retried ``judge_retries`` times; a reply without A, B or C
      is NOT_ATTEMPTED, as in simple-evals. Both are recorded as ``JUDGE_ERROR`` (``"call"``
      or ``"parse"``) and counted by ``judge_failures``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion.strip()
        truncated = is_truncated(state.output)
        md = {TRUNCATED: truncated, EXTRACTED: answer or None}
        if truncated or not answer:
            md[GRADE] = GRADES["C"]
            return Score(value=0.0, answer=answer, metadata=md,
                         explanation="not attempted: " + ("truncated" if truncated else "empty answer"))

        prompt = GRADER_TEMPLATE.format(question=state.metadata[question_key], target=target.text,
                                        predicted_answer=answer)
        grader = get_model(role=GRADER_ROLE, required=True, config=GenerateConfig(temperature=GRADER_TEMPERATURE))
        reply, error = "", "call"
        for attempt in range(1 + judge_retries):
            try:
                out = await grader.generate([ChatMessageSystem(content=GRADER_SYSTEM_MESSAGE),
                                             ChatMessageUser(content=prompt)])
                reply, error = out.completion, ""
                break
            except Exception:  # noqa: BLE001  (any failed call is retried)
                if attempt < judge_retries:
                    await asyncio.sleep(2 ** attempt)
        letter = read_grade(reply) if not error else None
        if not error and letter is None:
            error = "parse"
        if error:
            md[JUDGE_ERROR] = error
        md[GRADE] = GRADES[letter or "C"]
        return Score(value=1.0 if letter == "A" else 0.0, answer=answer, metadata=md,
                     explanation=f"{md[GRADE]}" + (f" (judge {error} failure)" if error else "")
                                 + f"; grader replied {reply[:200]!r}")

    return score
