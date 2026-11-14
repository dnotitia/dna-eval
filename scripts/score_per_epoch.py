"""
 ℹ️ 사용 방법 

    # logs 디렉토리 또는 .eval 파일 경로를 인자로 넘겨주면 됩니다. 여러 개 지정 가능합니다.
    $ python scripts/score_per_epoch.py logs/<log>.eval [more logs or directories...]

    # (Optional) --model-name 지정하면 score.log 파일에 찍힐 모델명을 지정할 수 있습니다.
    # (Optional) --out-dir 지정하면 score.log이 저장될 디렉토리를 지정할 수 있습니다.
    $ python scripts/score_per_epoch.py logs/ --model-name my-model --out-dir results/

    # (Optional) --task 지정 시 디렉토리 아래에서 해당 태스크만 찾아서 score.log에 찍히도록 합니다. 여러 개 지정 가능합니다.
    $ python scripts/score_per_epoch.py logs/ --task hallucination --task persona

    # 한국어 벤치마크 로그(kmmlu_pro, kbl 등)도 기록됩니다. 라벨에는 scorer 대신 태스크 이름이 들어갑니다.
    $ python scripts/score_per_epoch.py logs/ --task kmmlu_pro,kbl

For every run (epoch) of every DNA suite or benchmark log given, appends one line per score to
two files in ``--out-dir`` (default: the current directory), in the order the runs were made:

    score.log               planets.loc[len(planets)] = ["<label>", <mean score, 2 dp>, <passed>]
    score.python.list.log   ["<YYYYmmddHHMMSS>", "<model>", "<task>", <score sum>, <passed>, <samples>, <mean, 5 dp> ],

A run's mean is the mean of its samples' scores, failed calls and judgments included (they score
as the suite scores them), and passed counts the samples whose rubric threshold was met (for
benchmarks, see below). The timestamp is the time the run's last sample finished, in local time.
Labels per scorer:

    performance           <model>                     Performance
    persona               <model> (Pers)              Persona
    uncensorship          <model> (Unc)               Uncensorship
    hallucination         <model> (Hallu)             Hallucination
      per category        <model> (Hallu:<category>)  Hallucination:<category>
    language_confusion    <model> (Conf)              LanguageConfusion
    repetition            <model> (Rep)               Repetition
    reasoning_style       <model> (RStyle)            ReasoningStyle
    content_style         <model> (CStyle)            ContentStyle
    benchmark scorers     <model> (<task>)            <task>

Benchmark tasks share scorers (``letter_choice`` scores seven of them), so their rows are labelled
by the task's name (``kmmlu_pro``, ``kbl``, ...) instead. Their scorers give a sample 1 when it is
correct and 0 otherwise (wrong, truncated, without the answer format, not attempted, or not
judged), so passed counts the correct answers and mean is the run's accuracy.
``checklist_grade`` (``haerae_vision``) gives the fraction of checklist items met: mean is the
mean fraction, the task's own score, and passed counts the answers that met every item. Metrics
a task computes on top of the sample scores (``weighted_accuracy`` of ``kmmlu_pro``,
``macro_accuracy`` of ``kmmlu_hard``, the KoSimpleQA F-score, the KBL averages) are not recomputed
here; read them from the log. Task arguments are not part of the label: a run of some subsets
only, or of ``haerae_vision`` with ``question_type=explicit``, is labelled like a default run.

<model> is the evaluated model's name without its provider prefix (``vllm/my-model`` gives
``my-model``), unless ``--model-name`` is given. ``--task`` (repeatable, or comma-separated)
keeps only the logs of those tasks.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

# scorer name -> (score.log label suffix, task name in score.python.list.log)
LABELS = {
    "performance": ("", "Performance"),
    "persona": (" (Pers)", "Persona"),
    "uncensorship": (" (Unc)", "Uncensorship"),
    "hallucination": (" (Hallu)", "Hallucination"),
    "language_confusion": (" (Conf)", "LanguageConfusion"),
    "repetition": (" (Rep)", "Repetition"),
    "reasoning_style": (" (RStyle)", "ReasoningStyle"),
    "content_style": (" (CStyle)", "ContentStyle"),
}
GROUP_BY = {"hallucination": "category"}   # scorers whose scores are also reported per metadata value
# the scorers of the benchmark tasks (src/dna_eval/tasks/benchmarks/): reported under the task's name,
# since tasks share them; a sample passes when it scores 1
BENCHMARK_SCORERS = ("letter_choice", "math_answer", "kbl_answer", "simpleqa_grade", "answer_judge",
                     "checklist_grade")


def log_paths(targets: list[str]) -> list[str]:
    """The .eval/.json logs named, and those found in the directories named."""
    paths = []
    for t in targets:
        if Path(t).is_dir():
            paths += [info.name for info in list_eval_logs(t)]
        else:
            paths.append(t)
    return paths


def model_name(log: EvalLog) -> str:
    return log.eval.model.split("/")[-1]


def timestamp(iso: str | None) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y%m%d%H%M%S") if iso else ""


def run_lines(log: EvalLog, model: str) -> list[tuple[str, str, str, float, int, int]]:
    """(time, label, task, score sum, passed, samples) for each run and score, in run order."""
    name = log.eval.task.split("/")[-1]
    labels = LABELS | {scorer: (f" ({name})", name) for scorer in BENCHMARK_SCORERS}
    # (epoch, scorer, group) -> [(score, passed)], and each epoch's last finish time
    rows: dict[tuple[int, str, str | None], list[tuple[float, bool]]] = defaultdict(list)
    finished: dict[int, str] = {}
    for sample in log.samples or []:
        if sample.completed_at and sample.completed_at > finished.get(sample.epoch, ""):
            finished[sample.epoch] = sample.completed_at
        for scorer, score in (sample.scores or {}).items():
            if scorer not in labels:
                continue
            value = float(score.as_float())
            if scorer in BENCHMARK_SCORERS:
                passed = value >= 1
            else:
                passed = bool((score.metadata or {}).get("passed", False))
            entry = (value, passed)
            rows[(sample.epoch, scorer, None)].append(entry)
            if scorer in GROUP_BY:
                group = (sample.metadata or {}).get(GROUP_BY[scorer])
                rows[(sample.epoch, scorer, str(group))].append(entry)

    order = list(labels)
    keys = sorted(rows, key=lambda k: (k[0], order.index(k[1]), k[2] is not None, k[2] or ""))
    lines = []
    for epoch, scorer, group in keys:
        suffix, task = labels[scorer]
        if group is not None:
            suffix, task = f"{suffix[:-1]}:{group})", f"{task}:{group}"
        scores = rows[(epoch, scorer, group)]
        lines.append((timestamp(finished.get(epoch) or log.stats.completed_at), f"{model}{suffix}", task,
                      sum(s for s, _ in scores), sum(p for _, p in scores), len(scores)))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="eval log files, or directories of them")
    parser.add_argument("--model-name", help="the model name to write (default: from the log)")
    parser.add_argument("--out-dir", default=".", help="where score.log and score.python.list.log are")
    parser.add_argument("--task", action="append", default=[],
                        help="only the logs of this task (e.g. hallucination, kmmlu_pro); "
                             "repeatable or comma-separated")
    args = parser.parse_args()
    tasks = {t.strip().removeprefix("dna_eval/") for arg in args.task for t in arg.split(",") if t.strip()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in log_paths(args.logs):
        if tasks and read_eval_log(path, header_only=True).eval.task.split("/")[-1] not in tasks:
            continue
        log = read_eval_log(path)
        if log.status != "success":
            print(f"skipped {path}: status {log.status}", file=sys.stderr)
            continue
        model = args.model_name or model_name(log)
        lines = run_lines(log, model)
        with open(out / "score.log", "a") as score_log, open(out / "score.python.list.log", "a") as list_log:
            for when, label, task, total, passed, n in lines:
                score_log.write(f'planets.loc[len(planets)] = ["{label}", {total / n:.2f}, {passed}]\n')
                list_log.write(f'["{when}", "{model}", "{task}", {total}, {passed}, {n}, {total / n:.5f} ],\n')
        print(f"{path}: {len(lines)} lines ({log.eval.task}, {model})", file=sys.stderr)
        written += len(lines)
    if not written:
        print("nothing written: no successful DNA suite or benchmark log matched", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
