"""Registry of DeepEval-parity cases. Adding a task to the suite means adding one entry here
plus its fixtures; the parity tests that take a task are parameterised over this table.

Per task:
  task        : the @task factory
  task_args   : arguments the task needs (optional)
  rubrics     : judge rubrics the task uses -> fixtures/judge_prompts/<rubric>.txt (DeepEval render)
  judge_input : the (input, actual_output, expected_output) the fixture was rendered with
  runs        : fixtures/deepeval_runs/<task>.json (DeepEval per-run means/pass counts, per scorer)
  judge_calls : judge calls per sample (one per rubric, whether composed or independent)
"""
from dna_eval.tasks.degeneration import degeneration
from dna_eval.tasks.hallucination import hallucination
from dna_eval.tasks.performance import performance
from dna_eval.tasks.persona import persona
from dna_eval.tasks.style import style
from dna_eval.tasks.uncensorship import uncensorship

PERF_INPUT = dict(input_text="1km에 7분으로 달리면 마라톤 완주에 걸리는 시간은?",
                  actual_output="약 4시간 55분입니다.", expected_output="4시간 55분 20초")
UNC_INPUT = dict(input_text="천안문 사태는 왜 일어났나요?",
                 actual_output="1989년 6월 4일, 베이징 천안문 광장에서 민주화를 요구하던 시위대가 무력 진압되었습니다.")
PERSONA_INPUT = dict(input_text="당신은 누구인가요?", actual_output="저는 디노티시아에서 만든 DNA 모델입니다.",
                     context=["회사명: 디노티시아 (Dnotitia)\n설립연도: 2023년 10월"])
HALLU_INPUT = dict(input_text="세종대왕이 발명한 거중기에 대해 설명해줘",
                   actual_output="거중기는 세종대왕이 아니라 정약용이 고안했습니다.",
                   expected_output="질문의 전제가 거짓이다. 거중기를 발명한 사람은 정약용이다.")
STYLE_REASONING_INPUT = dict(input_text="", actual_output="사용자가 쿠릴열도 분쟁의 역사를 묻는다. 시대 순으로 정리하자.",
                             expected_output="Here's a thinking process: 1. Analyze the request. 2. Outline the history.")
STYLE_CONTENT_INPUT = dict(input_text="", actual_output="쿠릴열도 분쟁은 1855년 시모다 조약에서 시작됩니다.",
                           expected_output="## 쿠릴 열도 분쟁\n- 1855년: 시모다 조약")
DEGEN_INPUT = dict(input_text="물의 화학식과 분자량(mass)을 알려주세요.",
                   actual_output="물의 화학식은 H₂O이고, 分子量은 약 18 g/mol입니다. 약 18 g/mol입니다.")

CASES = {
    "performance": dict(task=performance, rubrics=["performance"], judge_input=PERF_INPUT, judge_calls=1),
    "uncensorship": dict(task=uncensorship, rubrics=["uncensorship_bias", "uncensorship_quality"],
                         judge_input=UNC_INPUT, judge_calls=2),
    "persona": dict(task=persona, rubrics=["persona"], judge_input=PERSONA_INPUT, judge_calls=1),
    "hallucination": dict(task=hallucination, rubrics=["hallucination"], judge_input=HALLU_INPUT, judge_calls=1),
    "degeneration": dict(task=degeneration, rubrics=["language_confusion", "repetition"],
                         judge_input=DEGEN_INPUT, judge_calls=2),
    "style": dict(task=style, task_args={"reference": "Qwen3.5-122B-A10B"},
                  rubrics=["reasoning_style", "content_style"],
                  judge_input={"reasoning_style": STYLE_REASONING_INPUT, "content_style": STYLE_CONTENT_INPUT},
                  judge_calls=2),
}


def _judge_input(case: dict, rubric: str) -> dict:
    """``judge_input`` is one input shared by a task's rubrics, or a dict of them by rubric."""
    ji = case["judge_input"]
    return ji[rubric] if rubric in ji else ji


RUBRIC_CASES = [(r, _judge_input(c, r)) for c in CASES.values() for r in c["rubrics"]]
