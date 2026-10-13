"""The DNA suite's conventions, shared by every task in this package.

One convention for every task, so a number from one task is comparable with another:

- how the model under evaluation is called: no system prompt, the sample's input as a single
  user turn, sampling left to the served model's ``generation_config``, a fixed token budget,
  and thinking toggled through vLLM's ``chat_template_kwargs``;
- what is reported first: the per-run trimmed mean (``per_run_trimmed``), not Inspect's
  default first metric (``mean``).

Changing these values moves every score in the suite at once; treat it as a suite version
bump, like editing ``prompts/geval_results.jinja``. Keep this module free of task imports so
the task modules can import from it.
"""
from inspect_ai.log import HeadlineMetric
from inspect_ai.model import GenerateConfig

MAX_TOKENS = 16384      # answer budget, reasoning included
MAX_RETRIES = 3         # retries of a failed request to the model under evaluation (Inspect's)

# The number Inspect shows first for a task (log header, log listing). Without it Inspect
# picks ``mean``, which it computes before the per-run metrics.
HEADLINE = HeadlineMetric(metric="trimmed_mean")


def dna_generate_config(thinking: bool = True, **overrides) -> GenerateConfig:
    """The suite's default ``GenerateConfig``; ``overrides`` win over the defaults.

    A task that overrides anything should say why in its own file.
    """
    base = dict(
        max_tokens=MAX_TOKENS,
        max_retries=MAX_RETRIES,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    base.update(overrides)
    return GenerateConfig(**base)
