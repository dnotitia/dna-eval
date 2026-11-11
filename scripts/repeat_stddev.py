"""
 ℹ️ 사용 방법

    # 같은 모델·같은 설정으로 반복한 평가의 로그 디렉토리를 회차 순서대로 넘겨줍니다 (회차마다 디렉토리 하나).
    $ python scripts/repeat_stddev.py logs/run-1 logs/run-2 logs/run-3 logs/run-4 logs/run-5

    # (Optional) --task 지정 시 해당 태스크만 계산합니다. 여러 번 지정하거나 쉼표로 이어 쓸 수 있습니다.
    $ python scripts/repeat_stddev.py logs/run-{1..5} --task performance,hallucination,persona

    # (Optional) --metric 지정 시 대표 지표와 같은 채점 항목 안에서 그 지표로 계산합니다.
    $ python scripts/repeat_stddev.py logs/run-{1..5} --metric mean

How much a task's score moves when the same evaluation is repeated. Each directory is one
evaluation; for every task found in them, the script reads the task's score from each directory
(the headline metric the log reports, e.g. ``trimmed_mean`` for the DNA evaluation suite, in %),
and prints the scores, their mean, and their sample standard deviation:

    sd = sqrt( sum_i (x_i - mean)^2 / (n - 1) )        n = number of directories

The last line is the largest standard deviation over the tasks. A directory holding more than
one successful log of a task uses the most recent one; a task without a successful log in every
directory is listed as missing and left out of the largest value (exit code 1).
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from inspect_ai.log import EvalLog, headline_metric, list_eval_logs, read_eval_log


def task_name(log: EvalLog) -> str:
    return log.eval.task.split("/")[-1]


def latest_successful(directory: str) -> dict[str, EvalLog]:
    """The most recent successful log header of each task in ``directory``."""
    found: dict[str, EvalLog] = {}
    for info in list_eval_logs(directory):
        log = read_eval_log(info.name, header_only=True)
        if log.status != "success":
            continue
        t = task_name(log)
        if t not in found or (log.eval.created or "") > (found[t].eval.created or ""):
            found[t] = log
    return found


def score(log: EvalLog, metric: str | None) -> tuple[str, float]:
    """Read the headline score's metric, keeping its scorer, score and reducer identity."""
    resolved = headline_metric(log)
    if resolved is None:
        raise ValueError(f"{task_name(log)}: no headline metric in evaluation results")
    values = resolved.score.metrics
    name = resolved.name if metric is None else metric
    if name not in values:
        raise ValueError(
            f"{task_name(log)}: scorer {resolved.score.scorer!r}, score {resolved.score.name!r}, "
            f"reducer {resolved.score.reducer!r} has no metric {name!r} (has {sorted(values)})"
        )
    try:
        value = float(values[name].value) * 100
    except OverflowError as exc:
        raise ValueError(f"{task_name(log)}: non-finite percentage for metric {name!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{task_name(log)}: non-finite percentage for metric {name!r}")
    return name, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("dirs", nargs="+", help="one log directory per repeated evaluation, in order")
    parser.add_argument("--task", action="append", default=[], help="only these tasks; repeatable or comma-separated")
    parser.add_argument("--metric", help="use this metric within each task's headline score")
    args = parser.parse_args()
    if len(args.dirs) < 2:
        parser.error("give at least two directories")
    wanted = {t.strip() for arg in args.task for t in arg.split(",") if t.strip()}

    runs = [latest_successful(d) for d in args.dirs]
    tasks = sorted(wanted or set().union(*runs))
    for i, d in enumerate(args.dirs, 1):
        print(f"#{i}: {d}")
    print(f"\n{'task':16} {'metric':18} " + " ".join(f"{'#' + str(i):>8}" for i in range(1, len(args.dirs) + 1))
          + f" {'mean':>8} {'sd(n-1)':>8}")
    sds: dict[str, float] = {}
    missing = []
    for t in tasks:
        if any(t not in r for r in runs):
            missing.append(t)
            print(f"{t:16} missing in: " + ", ".join(d for d, r in zip(args.dirs, runs) if t not in r))
            continue
        scored = []
        for directory, results in zip(args.dirs, runs):
            try:
                scored.append(score(results[t], args.metric))
            except ValueError as exc:
                parser.error(f"{directory}: {exc}")
        xs = [x for _, x in scored]
        sds[t] = statistics.stdev(xs)
        print(f"{t:16} {scored[0][0]:18} " + " ".join(f"{x:8.2f}" for x in xs)
              + f" {statistics.mean(xs):8.2f} {sds[t]:8.3f}")
    if sds:
        worst = max(sds, key=sds.get)
        print(f"\nlargest sd: {sds[worst]:.3f} ({worst}, {len(args.dirs)} evaluations)")
    return 1 if missing or not sds else 0


if __name__ == "__main__":
    sys.exit(main())
