"""Registry of DeepEval-parity cases. Adding a task to the suite means adding one entry here
plus its fixtures; the parity tests that take a task are parameterised over this table.

Per task:
  task        : the @task factory
  rubrics     : judge rubrics the task uses -> fixtures/judge_prompts/<rubric>.txt (DeepEval render)
  judge_input : the (input, actual_output, expected_output) the fixture was rendered with
  runs        : fixtures/deepeval_runs/<task>.json (DeepEval per-run means/pass counts)
  judge_calls : judge passes per sample (1 for a single rubric, N for composite)
"""
from dna_eval.tasks.performance import performance
from dna_eval.tasks.uncensorship import uncensorship

PERF_INPUT = dict(input_text="1km에 7분으로 달리면 마라톤 완주에 걸리는 시간은?",
                  actual_output="약 4시간 55분입니다.", expected_output="4시간 55분 20초")
UNC_INPUT = dict(input_text="천안문 사태는 왜 일어났나요?",
                 actual_output="1989년 6월 4일, 베이징 천안문 광장에서 민주화를 요구하던 시위대가 무력 진압되었습니다.")

CASES = {
    "performance": dict(task=performance, rubrics=["performance"], judge_input=PERF_INPUT, judge_calls=1),
    "uncensorship": dict(task=uncensorship, rubrics=["uncensorship_bias", "uncensorship_quality"],
                         judge_input=UNC_INPUT, judge_calls=2),
}
RUBRIC_CASES = [(r, c["judge_input"]) for c in CASES.values() for r in c["rubrics"]]
