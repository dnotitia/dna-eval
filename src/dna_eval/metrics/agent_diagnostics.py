"""``agent_diagnostics``: how agent runs ended, next to how they scored.

An agent task runs a loop until the agent finishes or a limit stops it (turns, steps, time).
A run stopped by a limit scores what its final state earns, usually 0; this metric reports how
often that happened and how long runs were, so a low score can be told apart from runs that
were cut short.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, SampleScore, Value, metric

# Keys an agent solver writes to ``state.store`` and its scorer copies into ``Score.metadata``.
LIMIT_REACHED = "limit_reached"   # the limit that ended the run ("turns", "steps", "time", ...), or None
TURNS = "turns"                   # model turns the agent took


@metric(scores="unreduced")
def agent_diagnostics() -> Metric:
    """``limit_reached``: the share of runs a limit ended; ``mean_turns``: the mean number of
    model turns. Every run counts once (``scores="unreduced"``)."""

    def compute(scores: list[SampleScore]) -> Value:
        md = [s.score.metadata or {} for s in scores]
        turns = [m[TURNS] for m in md if isinstance(m.get(TURNS), int)]
        return {
            "limit_reached": sum(bool(m.get(LIMIT_REACHED)) for m in md) / len(md) if md else 0.0,
            "mean_turns": sum(turns) / len(turns) if turns else float("nan"),
        }

    return compute
