#!/usr/bin/env bash
# Run a DNA evaluation suite task against one model.
#
#   bash run.sh <task> <model> <judge-model> [reference-model] [inspect eval options...]
#
#   <task>             performance, hallucination, persona, uncensorship, degeneration, style,
#                      or all (every task; style only when a reference model is given)
#   <model>            the model under evaluation, as Inspect names it (e.g. vllm/my-model)
#   <judge-model>      the judge, bound to the grader role (e.g. openai-api/judge/gemma-4-31B-it)
#   [reference-model]  the style task's reference model: required for style
#   [options...]       passed to `inspect eval` (e.g. -T thinking=false --log-dir logs/x)
#
# Endpoints and keys are read from the environment or a .env file (see README.md).
# Concurrency is fixed at --max-connections 20 per model unless the options set
# --max-connections, --adaptive-connections or --batch themselves.
set -euo pipefail

TASKS=(performance hallucination persona uncensorship degeneration style)

# Without --max-connections, Inspect scales concurrency on its own (up to 100 requests per model)
# and only scales down on rate-limit errors, which a self-hosted server rarely sends.
MAX_CONNECTIONS=20

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 3 ]] || usage
TASK=$1
MODEL=$2
JUDGE=$3
shift 3
REFERENCE=""
if [[ $# -gt 0 && $1 != -* ]]; then
  REFERENCE=$1
  shift
fi

concurrency=(--max-connections "$MAX_CONNECTIONS")
for arg in "$@"; do
  case $arg in
    --max-connections*|--adaptive-connections*|--batch*) concurrency=() ;;
  esac
done

run() {   # run <task> [options...]
  local task=$1
  shift
  inspect eval "dna_eval/$task" --model "$MODEL" --model-role "grader=$JUDGE" \
    ${concurrency[@]+"${concurrency[@]}"} "$@"
}

case $TASK in
  all)
    others=()
    for t in "${TASKS[@]}"; do
      [[ $t == style ]] || others+=("dna_eval/$t")
    done
    inspect eval "${others[@]}" --model "$MODEL" --model-role "grader=$JUDGE" \
      ${concurrency[@]+"${concurrency[@]}"} "$@"
    if [[ -n $REFERENCE ]]; then
      run style -T "reference=$REFERENCE" "$@"
    else
      echo "style skipped: give a reference model to run it" >&2
    fi
    ;;
  style)
    [[ -n $REFERENCE ]] || { echo "style needs a reference model" >&2; usage; }
    run style -T "reference=$REFERENCE" "$@"
    ;;
  *)
    [[ " ${TASKS[*]} " == *" $TASK "* ]] || { echo "unknown task: $TASK" >&2; usage; }
    [[ -z $REFERENCE ]] || { echo "only style takes a reference model" >&2; usage; }
    run "$TASK" "$@"
    ;;
esac
