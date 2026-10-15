"""``automationbench_state``: AutomationBench grading by the final state of the workspace.

Grading is the benchmark's own (``automationbench.rubric``, github.com/zapier/AutomationBench,
MIT), run on the world the agent left behind: each task lists assertions about the final
``WorldState``. Two numbers per run:

- ``task_completed_correctly``, the score value: 1 when every scored assertion holds, else 0.
  Its mean is the official AutomationBench pass rate;
- ``partial_credit``, in ``Score.metadata``: the share of scored assertions that hold. An
  assertion that already held in the initial state is left out of the count while it still holds
  (doing nothing earns nothing) and counts as a failure once the agent breaks it; the official
  function applies this rule and is called unchanged.

The world is read from ``state.store["automationbench_world"]`` (JSON written by
``solvers/automationbench_agent.py``, which this module does not import) and rebuilt with the
official ``WorldState``; the task's assertions and initial state come from the sample's
``metadata["info"]``. A run ended by the turn cap is scored on its final world. Assertion
handlers run in the official strict mode: a handler that raises makes the sample an error
(decision record 0020), not a score of 0; set ``AUTOMATIONBENCH_STRICT_ASSERTIONS=0`` for the
official lenient mode (failed assertion instead).

``Score.metadata``: ``partial_credit``, ``assertions_passed`` and ``assertions_scored``
(counts), ``failed_assertions`` (the types of the failed ones), and ``limit_reached`` and
``turns`` copied from the store for ``agent_diagnostics``.
"""
from __future__ import annotations

import copy

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS, agent_diagnostics
from dna_eval.metrics.partial_credit import PARTIAL_CREDIT, partial_credit

WORLD_KEY = "automationbench_world"   # ``state.store`` key the agent solver writes
INFO_KEY = "info"                     # ``state.metadata`` key holding the task's info dict


@scorer(metrics=[accuracy(), stderr(), partial_credit(), agent_diagnostics()])
def automationbench_state() -> Scorer:
    """Score 1 when every scored assertion of the task holds on the final world, else 0
    (see the module docstring for the metadata)."""

    async def score(state: TaskState, target: Target) -> Score:
        from automationbench.rubric import partial_credit as official_partial_credit
        from automationbench.rubric import task_completed_correctly
        from automationbench.runner import WorldState, strip_none_values

        info = copy.deepcopy(state.metadata[INFO_KEY])
        info["assertions"] = [strip_none_values(a) for a in info.get("assertions", [])]
        initial_state = strip_none_values(info.get("initial_state", {}))
        graded = {"info": info, "world": WorldState(**state.store.get(WORLD_KEY)),
                  "initial_state": initial_state}
        credit = official_partial_credit(graded)
        passed = task_completed_correctly(graded)
        results = [r for r in graded.get("_assertion_results", []) if not r["excluded"]]
        failed = [r["type"] for r in results if not r["passed"]]
        return Score(
            value=passed,
            explanation=f"{len(results) - len(failed)} of {len(results)} scored assertions hold",
            metadata={PARTIAL_CREDIT: credit, "assertions_passed": len(results) - len(failed),
                      "assertions_scored": len(results), "failed_assertions": failed,
                      LIMIT_REACHED: state.store.get(LIMIT_REACHED), TURNS: state.store.get(TURNS)})

    return score
