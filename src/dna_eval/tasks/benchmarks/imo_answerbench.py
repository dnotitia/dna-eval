"""IMO-AnswerBench: 400 olympiad short-answer problems, graded by the model bound to ``grader``.

Official sources, followed as written:

- data: ``imobench/answerbench_v2.csv`` in github.com/google-deepmind/superhuman (columns
  Problem ID, Problem, Short Answer, Category, Subcategory, Source), pinned to a commit and a
  SHA-256, downloaded once and cached;
- grader: the AnswerAutoGrader prompt of the paper "Towards Robust Mathematical Reasoning"
  (arXiv 2511.01846, Appendix A.5); ``scorers/imo_answer_grader.py`` says how it is applied;
- reporting: the paper averages 8 runs, so ``epochs`` is 8 and the score is pass@1 averaged
  over runs, overall and per Category.

Deviations, all recorded in the log:

- **Subject prompt.** The benchmark ships none; the problem is preceded by "Put your final answer
  within \\boxed{}.", so the grader can read the answer from the last box.
- **Grader prompt.** The paper's prompt refers to an extraction section it does not print. The
  section in ``prompts/imo_answer_grader.jinja`` is this project's addition (the answer is the
  last ``\\boxed{}``); the rest is verbatim.
- **Grader model.** The paper's grader is a Gemini model. Here it is whatever model the ``grader``
  role is bound to (required); numbers are comparable only to runs with the same grader.
- **Reasoning-model handling** (decision record 0016): thinking on, sampling from the served
  model's ``generation_config``, no ``max_tokens``, the grader sees only the final content; a
  cut-off answer scores 0 without a grader call and is reported by ``answer_diagnostics``.
"""
from __future__ import annotations

import csv
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import HeadlineMetric
from inspect_ai.solver import generate

from dna_eval.scorers.imo_answer_grader import imo_answer_grader
from dna_eval.tasks.benchmarks import benchmark_generate_config
from dna_eval.utils.cache import cache_root, fetch_pinned

REPO = "google-deepmind/superhuman"
CSV_PATH = "imobench/answerbench_v2.csv"
# Commit this task is pinned to, and the SHA-256 of the CSV at that commit (decision record 0008).
COMMIT = "80b2527a0b4e4bfc6a8b28825fadbdcfdd6048a1"
CSV_SHA256 = "275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9"

INSTRUCTION = "Put your final answer within \\boxed{}."


def fetch_csv(commit: str = COMMIT, sha256: str | None = CSV_SHA256) -> Path:
    """The answerbench CSV at ``commit``, downloaded once into the cache directory and checked
    against ``sha256``. ``sha256=None`` skips the check, for ``-T commit=<other commit>`` runs,
    which are therefore not checksum-pinned."""
    url = f"https://raw.githubusercontent.com/{REPO}/{commit}/{CSV_PATH}"
    return fetch_pinned(url, cache_root() / "benchmarks" / "imo_answerbench" / f"answerbench_v2.{commit}.csv", sha256)


def read_samples(path: Path) -> list[Sample]:
    """One ``Sample`` per CSV row. The input is the instruction followed by the problem; the
    bare problem is kept in ``metadata["problem"]`` for the grader."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Sample(id=r["Problem ID"], input=f"{INSTRUCTION}\n\n{r['Problem'].strip()}", target=r["Short Answer"].strip(),
               metadata={"problem": r["Problem"].strip(), "category": r["Category"],
                         "subcategory": r["Subcategory"], "source": r["Source"]})
        for r in rows]


@task
def imo_answerbench(thinking: bool = True, epochs: int = 8, commit: str = COMMIT) -> Task:
    """IMO-AnswerBench (400 problems) x ``epochs`` runs. Headline: ``accuracy`` (pass@1 averaged
    over runs); also ``stderr``, accuracy by ``category`` and ``answer_diagnostics``.

    ``thinking`` is forwarded as ``chat_template_kwargs.enable_thinking``. ``commit`` selects the
    benchmark repository commit; only the default is checked against the pinned SHA-256.
    Needs ``--model-role grader=<judge>``.
    """
    path = fetch_csv(commit, CSV_SHA256 if commit == COMMIT else None)
    return Task(
        dataset=MemoryDataset(read_samples(path), name="imo-answerbench"),
        solver=generate(),
        scorer=imo_answer_grader(),
        epochs=Epochs(epochs, "mean"),
        config=benchmark_generate_config(thinking),
        headline_metric=HeadlineMetric(metric="accuracy"),
    )
