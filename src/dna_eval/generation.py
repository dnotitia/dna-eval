"""How the suite calls the model under evaluation.

One convention for every task, so a number from one task is comparable with another:
no system prompt, the sample's input as a single user turn, sampling left to the served
model's ``generation_config``, a fixed token budget, and thinking toggled
through vLLM's ``chat_template_kwargs``.

Changing these values moves every score in the suite at once; treat it as a suite
version bump, like editing ``prompts/geval_results.jinja``.
"""
from inspect_ai.model import GenerateConfig

MAX_TOKENS = 16384      # answer budget, reasoning included
MAX_RETRIES = 3         # attempts per call to the model under evaluation


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
