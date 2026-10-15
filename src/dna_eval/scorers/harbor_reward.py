"""``harbor_reward``: the reward a Harbor task's own verifier gives, as a score.

A Harbor task (Terminal-Bench, SWE-bench, DeepSWE, ...) ships its checking code: a test script
that runs in the sandbox after the agent has finished and writes a reward to
``/logs/verifier/reward.txt`` or ``reward.json``. ``inspect_harbor.harbor_scorer`` runs it and
reads the reward; this scorer wraps it and follows Harbor's reading of the result:

- the score is the reward as a float (0 or 1 for the benchmarks used here; the other numbers a
  ``reward.json`` holds are kept in ``Score.metadata["reward_dict"]``);
- a negative reward is the verifier's own crash marker (DeepSWE writes ``-1`` when its grader
  fails before it can grade), not a failed attempt: it raises, so the sample is an error that
  ``--retry-on-error`` can repeat, in line with decision record 0020;
- a verifier that times out, cannot be copied into the sandbox, or leaves no readable reward
  raises in ``harbor_scorer`` and stays an error, for the same reason.

``Score.metadata`` also carries what the agent solver left in ``state.store`` for
``agent_diagnostics`` (``limit_reached``, ``turns``), and the dataset (with its digest) and
the scaffold the task was run with.
"""
from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from dna_eval.metrics.agent_diagnostics import LIMIT_REACHED, TURNS, agent_diagnostics

DATASET = "dataset"     # Score.metadata: the pinned Harbor dataset, ``org/name@sha256:<digest>``
SCAFFOLD = "scaffold"   # Score.metadata: the agent loop the task was run with


def _inner_scorer(default_verifier_timeout_sec: int) -> Scorer:
    """``inspect_harbor``'s scorer (imported here, so this module imports without it)."""
    try:
        from inspect_harbor import harbor_scorer
    except ImportError as e:
        raise RuntimeError('this scorer needs the "inspect_harbor" package: pip install "dna-eval[agentic]"') from e
    return harbor_scorer(default_verifier_timeout_sec=default_verifier_timeout_sec)


@scorer(metrics=[accuracy(), stderr(), agent_diagnostics()])
def harbor_reward(dataset: str = "", scaffold: str = "", default_verifier_timeout_sec: int = 600) -> Scorer:
    """The reward of the task's verifier: 1 when the task's tests pass, 0 when they do not.

    ``default_verifier_timeout_sec`` applies to a task that sets no ``[verifier].timeout_sec``.
    ``dataset`` and ``scaffold`` are only recorded in ``Score.metadata``, next to
    ``limit_reached`` and ``turns`` (copied from ``state.store``), ``reward_dict`` (when the
    verifier wrote ``reward.json``) and ``Score.answer`` (``PASS``/``FAIL``).

    Raises ``RuntimeError`` for a negative reward (the verifier's crash marker); errors of
    ``harbor_scorer`` itself (timeout, missing or unreadable reward, copy failure) propagate.
    """
    inner = _inner_scorer(default_verifier_timeout_sec)

    async def score(state: TaskState, target: Target) -> Score:
        result = await inner(state, target)
        reward = float(result.value)  # harbor_scorer always returns the reward as a float
        if reward < 0:
            raise RuntimeError(f"the verifier reported a crash (reward {reward:g}): {result.explanation}")
        metadata = dict(result.metadata or {})
        metadata |= {LIMIT_REACHED: state.store.get(LIMIT_REACHED), TURNS: state.store.get(TURNS),
                     DATASET: dataset, SCAFFOLD: scaffold}
        return Score(value=reward, answer=result.answer, explanation=result.explanation, metadata=metadata)

    return score
