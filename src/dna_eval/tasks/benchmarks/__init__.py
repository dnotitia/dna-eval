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
from inspect_ai.log import HeadlineMetric
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Metric, accuracy, grouped, stderr

from dna_eval.metrics.answer_diagnostics import answer_diagnostics

MAX_RETRIES = 3         # retries of a failed request to the model under evaluation

# The number Inspect shows first, unless a task's benchmark leads with another one.
HEADLINE = HeadlineMetric(metric="accuracy")


def benchmark_generate_config(thinking: bool = True, **overrides) -> GenerateConfig:
    """``GenerateConfig`` for a benchmark task: thinking toggled through vLLM's
    ``chat_template_kwargs``, no ``max_tokens``. ``overrides`` win over the defaults; a task
    that overrides anything says why in its own file."""
    base = dict(max_retries=MAX_RETRIES, extra_body={"chat_template_kwargs": {"enable_thinking": thinking}})
    base.update(overrides)
    return GenerateConfig(**base)


def benchmark_metrics(*groups: str) -> list[Metric]:
    """``accuracy``, ``stderr`` and ``answer_diagnostics``, plus ``accuracy`` for each value of
    each sample-metadata field in ``groups``, reported as ``<field>/<value>``.

    A task's ``metrics`` replace its scorer's, so every benchmark task lists its metrics with
    this function; a task adds the benchmark's own numbers around it.
    """
    return [accuracy(), stderr(), answer_diagnostics(),
            *(grouped(accuracy(), g, all=False, name_template=f"{g}/{{group_name}}") for g in groups)]
