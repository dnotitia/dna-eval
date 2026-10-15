"""AutomationBench (Zapier), public split: business workflows an agent must carry out with tools.

Official source (github.com/zapier/AutomationBench, MIT, version 1.0.6), used as it is: the
tasks (``automationbench.domains``), the simulated workspace and its tools
(``automationbench.schema`` and ``automationbench.tools``) and the grading
(``automationbench.rubric``). Each task gives the agent a request (the official system and user
messages) and a workspace in a seeded state (mail, CRM, calendar, spreadsheets, chat, ...
across 47 simulated services); it is graded by assertions on the workspace the agent leaves.

- data: the 600 public tasks, 100 in each of ``sales``, ``marketing``, ``operations``,
  ``support``, ``finance`` and ``hr``, shipped inside the package (the ``simple`` domain is a
  baseline outside the benchmark score and is not offered);
- agent loop (``solvers/automationbench_agent.py``): the model calls tools until it replies with
  none, at most ``max_turns`` (50, the official ``--max-steps``) times; the toolset is
  ``api`` (the official CLI default), ``zapier`` or ``limited_zapier``;
- grading (``scorers/automationbench_state.py``): ``task_completed_correctly`` (every assertion
  holds) is the score, so ``accuracy`` is the official pass rate; the mean ``partial_credit``
  (share of assertions that hold) is reported next to it.

Install (not on PyPI, requires Python 3.13): see ``INSTALL``. ``verifiers`` is pinned because
its latest release no longer provides the ``datasets`` and ``agents`` modules that the benchmark
imports; the pin is the one in the benchmark's own lock file.

The AutomationBench leaderboard scores a private held-out set of the same shape, and the
published public-set pass rates come from the benchmark's own runner: public-split numbers
measured here are directional, not comparable with the leaderboard.

Deviations, all recorded in the log (the agent-loop ones are listed in
``solvers/automationbench_agent.py``):

- **Scaffold.** The agent loop runs on Inspect's tool calling instead of the benchmark's
  ``verifiers`` environment, and the provider-specific request options its CLI sets (reasoning
  effort, a 64,000 token cap for some providers, prompt caching, batch transport) are not used.
  The tools, the world, the turn cap and the grading are the official ones.
- **Reasoning-model handling** (decision records 0016 and 0020): thinking on, sampling from the
  served model's ``generation_config``, no ``max_tokens``. A run the turn cap ends is scored on
  its final state and counted by ``agent_diagnostics``.
- ``epochs`` defaults to 1, as in the official runner.
"""
from __future__ import annotations

import json

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import accuracy, grouped, stderr

from dna_eval.metrics.agent_diagnostics import agent_diagnostics
from dna_eval.metrics.partial_credit import partial_credit
from dna_eval.scorers.automationbench_state import automationbench_state
from dna_eval.solvers.automationbench_agent import automationbench_agent
from dna_eval.tasks.agentic import HEADLINE, agentic_generate_config, require

# AutomationBench has no release tags and is not on PyPI: the commit the numbers were measured
# at (version 1.0.6), and ``verifiers==0.2.1``, the release that still provides the ``datasets``
# and ``agents`` modules the benchmark imports (its latest release does not).
COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
INSTALL = (f'pip install "automation-bench @ git+https://github.com/zapier/AutomationBench@{COMMIT}" '
           '"verifiers==0.2.1"')

PUBLIC_DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")


_MESSAGES = {"system": ChatMessageSystem, "user": ChatMessageUser, "assistant": ChatMessageAssistant}


def load_samples(domains: tuple[str, ...] = PUBLIC_DOMAINS) -> list[Sample]:
    """One sample per task of ``domains``.

    The input is the official prompt (system and user messages). The sample id is the task name
    (``sales.multi_hop_lookup``), and ``None`` values are dropped from ``info`` as the official
    runner does. ``metadata`` holds ``domain``, ``task_name``, ``example_id``
    and ``info``: the task's ``initial_state``, ``assertions`` and ``zapier_tools``, which the
    agent solver and the scorer read.
    """
    from automationbench.domains import get_domain_dataset
    from automationbench.runner import strip_none_values

    samples = []
    for domain in domains:
        for row in get_domain_dataset(domain):
            info = strip_none_values(json.loads(row["info"]))
            messages: list[ChatMessage] = [_MESSAGES[m["role"]](content=m["content"]) for m in row["prompt"]]
            samples.append(Sample(
                id=info["task_name"], input=messages, target=row["answer"],
                metadata={"domain": domain, "task_name": info["task_name"],
                          "example_id": row["example_id"], "info": info}))
    return samples


@task
def automationbench(domain: str | None = None, toolset: str = "api", thinking: bool = True,
                    epochs: int = 1, max_turns: int = 50) -> Task:
    """AutomationBench public split: 600 tasks, or the 100 of one ``domain``. Headline:
    ``accuracy`` (the official pass rate).

    ``domain`` is ``sales``, ``marketing``, ``operations``, ``support``, ``finance`` or ``hr``
    (default: all six). ``toolset`` is ``api`` (default), ``zapier`` or ``limited_zapier``.
    ``max_turns`` caps the model's replies per task. ``thinking`` is forwarded as
    ``chat_template_kwargs.enable_thinking``. Reports ``accuracy`` and ``stderr``, the mean
    ``partial_credit``, ``agent_diagnostics`` (share of runs the turn cap ended, mean turns) and
    ``accuracy`` by domain as ``domain/<name>``. Needs AutomationBench installed (``INSTALL``)
    and a model served with tool calling (``docs/agentic.md``).
    """
    require("automationbench", python=(3, 13), install=INSTALL)
    if domain is not None and domain not in PUBLIC_DOMAINS:
        raise ValueError(f"domain must be one of {', '.join(PUBLIC_DOMAINS)} (or omitted for all), got {domain!r}")
    return Task(
        dataset=MemoryDataset(load_samples(PUBLIC_DOMAINS if domain is None else (domain,)),
                              name="automationbench" if domain is None else f"automationbench-{domain}"),
        solver=automationbench_agent(toolset=toolset, max_turns=max_turns),
        scorer=automationbench_state(),
        metrics=[accuracy(), stderr(), partial_credit(), agent_diagnostics(),
                 grouped(accuracy(), "domain", all=False, name_template="domain/{group_name}")],
        epochs=Epochs(epochs, "mean"),
        config=agentic_generate_config(thinking),
        headline_metric=HEADLINE,
    )
