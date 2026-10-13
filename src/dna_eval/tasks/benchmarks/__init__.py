"""Conventions shared by the standard benchmark tasks in this package (decision record 0016).

These tasks follow each benchmark's official protocol and are compared with what other
model reports publish; they carry no DeepEval parity (0001). What they share:

- the model under evaluation is called with thinking on, sampling left to the served model's
  ``generation_config``, and no output cap: a cut-off answer would be scored wrong for a
  reason that has nothing to do with the benchmark;
- answers are read from the final content, never from the reasoning;
- a truncated answer or one without the required answer format scores 0 and is counted by
  ``metrics/answer_diagnostics.py``, so both rates are reported next to the accuracy.

Keep this module free of task imports so the task modules can import from it.
"""
from inspect_ai.model import GenerateConfig

MAX_RETRIES = 3         # retries of a failed request to the model under evaluation


def benchmark_generate_config(thinking: bool = True, **overrides) -> GenerateConfig:
    """``GenerateConfig`` for a benchmark task: thinking toggled through vLLM's
    ``chat_template_kwargs``, no ``max_tokens``. ``overrides`` win over the defaults; a task
    that overrides anything says why in its own file."""
    base = dict(max_retries=MAX_RETRIES, extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    base.update(overrides)
    return GenerateConfig(**base)
