#!/usr/bin/env bash
# Run DNA evaluation suite and Korean benchmark tasks against one model.
#
#   bash run.sh <task> <model> [judge-model] [reference-model] [inspect eval options...]
#
#   <task>             a task name, or a group (- and _ are interchangeable):
#                        dna_eval    the DNA evaluation suite: performance, hallucination, persona,
#                                    degeneration, style
#                        benchmarks  the Korean benchmarks: kmmlu_pro, kmmlu_redux, kmmlu_hard, click,
#                                    kobalt, hrm8k_ksm, haerae_bench, kosimpleqa, kormedmcqa, kbl,
#                                    haerae_vision, kmmmu
#                                    (haerae_vision and kmmmu need a vision-language model)
#                        all         every task of both groups
#                        text_only   every task but haerae_vision and kmmmu
#                        vision      haerae_vision and kmmmu
#                      style runs only when a reference model is given; a group skips it otherwise.
#                      uncensorship (DNA evaluation suite, private dataset) belongs to no group:
#                      it runs only when named.
#   <model>            the model under evaluation, as Inspect names it (e.g. vllm/my-model)
#   [judge-model]      the judge, bound to the grader role (e.g. openai-api/judge/gemma-4-31B-it):
#                      required by every group and every task but kmmlu_pro, kmmlu_redux,
#                      kmmlu_hard, click, kobalt, hrm8k_ksm, haerae_bench, kormedmcqa and kbl
#   [reference-model]  the style task's reference model: required for style
#   [options...]       passed to `inspect eval` (e.g. -T thinking=false --log-dir logs/x)
#
# Endpoints and keys are read from the environment or a .env file (see README.md).
set -euo pipefail

SUITE_TASKS=(performance hallucination persona degeneration style)
BENCHMARK_TASKS=(kmmlu_pro kmmlu_redux kmmlu_hard click kobalt hrm8k_ksm haerae_bench kosimpleqa kormedmcqa kbl
                 haerae_vision kmmmu)
VISION_TASKS=(haerae_vision kmmmu)   # need a vision-language model
NAMED_ONLY_TASKS=(uncensorship)      # in no group: run only when named
NO_JUDGE_TASKS=(kmmlu_pro kmmlu_redux kmmlu_hard click kobalt hrm8k_ksm haerae_bench kormedmcqa kbl)
                                     # score without a judge: the judge model may be left out

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 2 ]] || usage
TASK=${1//-/_}
MODEL=$2
shift 2
JUDGE=""
if [[ $# -gt 0 && $1 != -* ]]; then
  JUDGE=$1
  shift
fi
REFERENCE=""
if [[ $# -gt 0 && $1 != -* ]]; then
  REFERENCE=$1
  shift
fi
OPTIONS=("$@")

require_judge() {   # require_judge <task>...: stop with the usage when a task needs the judge left out
  [[ -z $JUDGE ]] || return 0
  local t
  for t in "$@"; do
    if [[ " ${NO_JUDGE_TASKS[*]} " != *" $t "* ]]; then
      echo "$TASK needs a judge model" >&2
      usage
    fi
  done
}

evaluate() {   # evaluate <task>... -- [options...]
  local tasks=()
  while [[ $1 != -- ]]; do
    tasks+=("dna_eval/$1")
    shift
  done
  shift
  local judge=()
  [[ -z $JUDGE ]] || judge=(--model-role "grader=$JUDGE")
  inspect eval "${tasks[@]}" --model "$MODEL" ${judge[@]+"${judge[@]}"} "$@"
}

run_group() {   # run_group <task>...: every task but style in one run, then style with the reference
  local others=() with_style=false
  for t in "$@"; do
    if [[ $t == style ]]; then with_style=true; else others+=("$t"); fi
  done
  require_judge "$@"
  evaluate "${others[@]}" -- ${OPTIONS[@]+"${OPTIONS[@]}"}
  if $with_style; then
    if [[ -n $REFERENCE ]]; then
      evaluate style -- -T "reference=$REFERENCE" ${OPTIONS[@]+"${OPTIONS[@]}"}
    else
      echo "style skipped: give a reference model to run it" >&2
    fi
  fi
}

case $TASK in
  all)        run_group "${SUITE_TASKS[@]}" "${BENCHMARK_TASKS[@]}" ;;
  dna_eval)   run_group "${SUITE_TASKS[@]}" ;;
  text_only)
    text=()
    for t in "${SUITE_TASKS[@]}" "${BENCHMARK_TASKS[@]}"; do
      [[ " ${VISION_TASKS[*]} " == *" $t "* ]] || text+=("$t")
    done
    run_group "${text[@]}"
    ;;
  vision)
    [[ -z $REFERENCE ]] || { echo "the vision tasks take no reference model" >&2; usage; }
    run_group "${VISION_TASKS[@]}"
    ;;
  benchmarks)
    [[ -z $REFERENCE ]] || { echo "the benchmarks take no reference model" >&2; usage; }
    run_group "${BENCHMARK_TASKS[@]}"
    ;;
  style)
    require_judge style
    [[ -n $REFERENCE ]] || { echo "style needs a reference model" >&2; usage; }
    evaluate style -- -T "reference=$REFERENCE" ${OPTIONS[@]+"${OPTIONS[@]}"}
    ;;
  *)
    [[ $TASK != *" "* && " ${SUITE_TASKS[*]} ${BENCHMARK_TASKS[*]} ${NAMED_ONLY_TASKS[*]} " == *" $TASK "* ]] ||
      { echo "unknown task: $TASK" >&2; usage; }
    [[ -z $REFERENCE ]] || { echo "only style takes a reference model" >&2; usage; }
    require_judge "$TASK"
    evaluate "$TASK" -- ${OPTIONS[@]+"${OPTIONS[@]}"}
    ;;
esac
