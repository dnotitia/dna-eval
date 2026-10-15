"""tau-bench (tau^3): customer-service agents tested against a simulated user, 4 text domains.

Official source: tau2-bench v1.0.1 (github.com/sierra-research/tau2-bench, MIT; papers
arXiv:2406.12045, 2506.07982, 2603.04370), whose environments, tools, policies, tasks, user
simulator and evaluator run as they are (``tau2_env.py``, ``solvers/tau_conversation.py``,
``scorers/tau_reward.py``). Domains and ``base`` task counts: ``airline`` 50, ``retail`` 114,
``telecom`` 114 (the user also has tools), ``banking_knowledge`` 97 (the agent searches a
knowledge base). One sample is one task; ``epochs`` are the trials of the official protocol
(``num_trials`` 4 for the leaderboard). Needs ``--model-role user=<model>`` for the simulated user and,
for the 40 retail tasks and 1 banking task with natural-language assertions,
``--model-role grader=<model>``.

Install tau2 with the command in ``tau2_env.INSTALL_COMMAND`` (Python 3.12 or 3.13). The data
the domains read (about 140 MB) is downloaded once from the pinned commit into the cache
directory.

Reported: ``accuracy`` (pass^1, the mean reward over trials) with its ``stderr``, ``pass_hat_k``
(pass^k for k = 1 ... ``epochs``, as tau2-bench computes it) and ``agent_diagnostics`` (the share of
runs a step, error or context limit ended, and the mean number of agent turns). Headline: ``accuracy``.

Deviations from the official protocol, all recorded in the log:

- **Agent loop and models.** The agent is the model under evaluation, driven through Inspect
  with the official prompt and tool schemas (``title`` keys dropped, ``$ref`` inlined; see
  ``solvers/tau_conversation.py`` for that and for the rare endings tau2 does not define). The user
  simulator is the ``user`` role at temperature 0 as officially, but it is whatever model you bind;
  the leaderboard uses frontier API models, and a different user model changes the numbers.
- **Reasoning-model handling** (decision records 0016, 0020): thinking on, sampling from the
  served model's ``generation_config`` (the official agent runs at temperature 0), no ``max_tokens``.
  The official limits apply: 200 steps, 10 failed tool calls.
- **Judge.** The official natural-language-assertion judge is ``gpt-4.1``; here it is the ``grader``
  role (see ``scorers/tau_reward.py``).
- **banking_knowledge retrieval.** The leaderboard's ``alltools`` variant needs an embedding API
  and a sandbox shell. ``retrieval_variant`` defaults to ``bm25`` (BM25 search over the knowledge
  (``KB_search``), no external service; needs ``rank-bm25``); the other
  offline variants are ``bm25_grep``, ``grep_only``, ``full_kb``, ``golden_retrieval`` and
  ``no_knowledge``. Scores are comparable only between runs with the same variant. Other domains
  ignore it.
- Trials are not seeded.
"""
from __future__ import annotations

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import accuracy, stderr

from dna_eval import tau2_env
from dna_eval.metrics.agent_diagnostics import agent_diagnostics
from dna_eval.metrics.pass_hat_k import pass_hat_k
from dna_eval.scorers.tau_reward import tau_reward
from dna_eval.solvers.tau_conversation import tau_conversation
from dna_eval.tasks.agentic import HEADLINE, agentic_generate_config, require

DEFAULT_RETRIEVAL_VARIANT = "bm25"


def load_samples(domain: str, split: str, retrieval_variant: str) -> list[Sample]:
    """One sample per tau2 task of ``domain`` in ``split``. The input is the user simulator's
    scenario (the solver builds the real conversation); the metadata names the task for the
    solver and the scorer."""
    tau2_env.tau2_ready()
    from tau2.data_model.tasks import RewardType

    samples = []
    for t in tau2_env.load_tasks(domain, split):
        criteria = t.evaluation_criteria
        samples.append(Sample(
            id=t.id, input=str(t.user_scenario),
            metadata={"domain": domain, "split": split, "task_id": t.id, "retrieval_variant": retrieval_variant,
                      "reward_basis": [r.value for r in criteria.reward_basis] if criteria else [],
                      "has_nl_assertions": bool(criteria and RewardType.NL_ASSERTION in criteria.reward_basis
                                                and criteria.nl_assertions)}))
    return samples


@task
def tau_bench(domain: str = "airline", split: str = "base", thinking: bool = True, epochs: int = 4,
              max_steps: int = 200, max_errors: int = 10,
              retrieval_variant: str = DEFAULT_RETRIEVAL_VARIANT) -> Task:
    """tau-bench ``domain`` (``airline``, ``retail``, ``telecom`` or ``banking_knowledge``), ``epochs``
    trials per task. Headline: ``accuracy``. ``thinking`` is forwarded as
    ``chat_template_kwargs.enable_thinking``. Needs ``--model-role user=<model>`` (and ``grader`` for
    tasks with natural-language assertions)."""
    require("tau2", python=(3, 12), install=tau2_env.INSTALL_COMMAND)
    if domain not in tau2_env.DOMAINS:
        raise ValueError(f"unknown tau-bench domain {domain!r}; choose from {', '.join(tau2_env.DOMAINS)}")
    tau2_env.check_retrieval_variant(domain, retrieval_variant)
    tau2_env.tau2_ready()
    return Task(
        dataset=MemoryDataset(load_samples(domain, split, retrieval_variant), name=f"tau-bench-{domain}"),
        solver=tau_conversation(max_steps=max_steps, max_errors=max_errors),
        scorer=tau_reward(),
        metrics=[accuracy(), stderr(), pass_hat_k(), agent_diagnostics()],
        epochs=Epochs(epochs, "mean"),
        config=agentic_generate_config(thinking),
        headline_metric=HEADLINE,
    )
