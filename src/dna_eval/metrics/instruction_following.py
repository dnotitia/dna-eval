"""``instruction_accuracy``: the four accuracies of the instruction-following benchmarks.

IFEval and IFBench report the same four numbers, each under a *strict* and a *loose* check of
the response (``scorers/instruction_following.py`` runs both):

- **prompt-level**: the share of prompts for which *every* instruction was followed;
- **instruction-level**: the share of all instructions that were followed (a micro average:
  a prompt with three instructions weighs three times a prompt with one).

``@metric(scores="unreduced")`` hands this metric every (sample, epoch) score. Each one counts
once, so with several epochs the numbers are pooled over all of them, not averaged per sample
(an average of per-sample fractions would weigh a prompt's instructions differently across
prompts). For the prompt-level numbers the two agree; for the instruction-level numbers they
differ whenever prompts have different numbers of instructions.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Score.metadata keys the instruction-following scorer sets for this metric.
STRICT = "strict"        # list[bool]: the strict check of each instruction of the prompt
LOOSE = "loose"          # list[bool]: the loose check of each instruction of the prompt
TRUNCATED = "truncated"  # bool: generation hit a token or context limit


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


@metric(scores="unreduced")
def instruction_accuracy() -> Metric:
    """Pooled accuracies over every scored (sample, epoch):

    - ``prompt_level_strict``, ``prompt_level_loose``, ``instruction_level_strict``,
      ``instruction_level_loose``: the four official numbers;
    - ``truncated``: the share of responses cut off at a token or context limit, which count
      as following no instruction;
    - ``prompt_level_strict_excl_truncated``, ``prompt_level_loose_excl_truncated``: the prompt-level
      numbers over the responses that were not cut off.
    """

    def compute(scores: list[SampleScore]) -> Value:
        rows = []                               # (strict, loose, truncated) per scored (sample, epoch)
        for s in scores:
            md = s.score.metadata or {}
            rows.append((list(md.get(STRICT, [])), list(md.get(LOOSE, [])), bool(md.get(TRUNCATED))))
        kept = [r for r in rows if not r[2]]

        def prompt_level(rs: list, i: int) -> float:
            return _mean([float(all(r[i])) for r in rs])

        def instruction_level(rs: list, i: int) -> float:
            return _mean([float(ok) for r in rs for ok in r[i]])

        return {
            "prompt_level_strict": prompt_level(rows, 0),
            "prompt_level_loose": prompt_level(rows, 1),
            "instruction_level_strict": instruction_level(rows, 0),
            "instruction_level_loose": instruction_level(rows, 1),
            "truncated": _mean([float(r[2]) for r in rows]),
            "prompt_level_strict_excl_truncated": prompt_level(kept, 0),
            "prompt_level_loose_excl_truncated": prompt_level(kept, 1),
        }

    return compute
