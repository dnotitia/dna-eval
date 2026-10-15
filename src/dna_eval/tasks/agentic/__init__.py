"""Conventions shared by the agent tasks in this package (decision record 0020).

These tasks run the model under evaluation in a loop: it calls tools, and in some tasks talks
with a simulated user, until it finishes or a limit stops it. What they share:

- the benchmark's own environment and scoring code are called as they are, imported inside the
  task, solver or scorer function, so dna-eval installs without them;
- the model is called with thinking on, sampling left to the served model's
  ``generation_config``, and no ``max_tokens``; turn, step and time limits follow the
  benchmark's official settings, and a run a limit ends is scored on its final state and
  counted by ``metrics/agent_diagnostics.py``;
- infrastructure failures (a sandbox or verifier that breaks) are sample errors, retried with
  ``--retry-on-error``, not scores of 0;
- a simulated user is the model bound to the ``user`` role, a judge the one bound to ``grader``,
  both required;
- where a container is needed the task declares a Docker-compatible sandbox; which provider
  runs it (``docker``, ``k8s``, cloud providers) is the user's choice (``--sandbox``).

Keep this module free of task imports so the task modules can import from it.
"""
from __future__ import annotations

from inspect_ai.log import HeadlineMetric
from inspect_ai.model import GenerateConfig

from dna_eval.utils.optional import require

__all__ = ["HEADLINE", "MAX_RETRIES", "agentic_generate_config", "require"]

MAX_RETRIES = 3         # retries of a failed request to the model under evaluation

# The number Inspect shows first, unless a task's benchmark leads with another one.
HEADLINE = HeadlineMetric(metric="accuracy")


def agentic_generate_config(thinking: bool = True, **overrides) -> GenerateConfig:
    """``GenerateConfig`` for an agent task: thinking toggled through vLLM's
    ``chat_template_kwargs``, no ``max_tokens``. ``overrides`` win over the defaults; a task
    that overrides anything says why in its own file."""
    base = dict(max_retries=MAX_RETRIES, extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    base.update(overrides)
    return GenerateConfig(**base)
