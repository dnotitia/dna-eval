"""Entry point Inspect imports for this package: everything that can be looked up by name.

``inspect eval dna_eval/<task>`` and ``inspect score --scorer dna_eval/<rubric>`` resolve
through the modules imported here. Importing ``scorers.rubric_scorer`` registers one scorer
per rubric in ``rubrics/``, whether or not a task uses it.
"""
from dna_eval.metrics import (agent_diagnostics, answer_diagnostics, calibration_error,  # noqa: F401
                              call_failures, instruction_following, judge_failures, license_pass,
                              partial_credit, pass_hat_k, per_run_trimmed, sub_step_accuracy,
                              weighted_accuracy)
from dna_eval.scorers import (automationbench_state, hle_judge, imo_answer_grader,  # noqa: F401
                              letter_choice, math_answer, rubric_scorer, scicode_tests, tau_reward)
from dna_eval.scorers import instruction_following as _instruction_following  # noqa: F401
from dna_eval.solvers import (automationbench_agent, dna_generate, scicode_steps,  # noqa: F401
                              shuffled_choices, style_reference, tau_conversation)
from dna_eval.tasks.agentic.automationbench import automationbench  # noqa: F401
from dna_eval.tasks.agentic.tau_bench import tau_bench  # noqa: F401
from dna_eval.tasks.benchmarks.aime2025 import aime2025  # noqa: F401
from dna_eval.tasks.benchmarks.aime2026 import aime2026  # noqa: F401
from dna_eval.tasks.benchmarks.click import click  # noqa: F401
from dna_eval.tasks.benchmarks.gpqa_diamond import gpqa_diamond  # noqa: F401
from dna_eval.tasks.benchmarks.hle import hle  # noqa: F401
from dna_eval.tasks.benchmarks.hrm8k_ksm import hrm8k_ksm  # noqa: F401
from dna_eval.tasks.benchmarks.ifbench import ifbench  # noqa: F401
from dna_eval.tasks.benchmarks.ifeval import ifeval  # noqa: F401
from dna_eval.tasks.benchmarks.imo_answerbench import imo_answerbench  # noqa: F401
from dna_eval.tasks.benchmarks.kmmlu_pro import kmmlu_pro  # noqa: F401
from dna_eval.tasks.benchmarks.kobalt import kobalt  # noqa: F401
from dna_eval.tasks.benchmarks.mmlu_pro import mmlu_pro  # noqa: F401
from dna_eval.tasks.benchmarks.scicode import scicode  # noqa: F401
from dna_eval.tasks.dna_suite.degeneration import degeneration  # noqa: F401
from dna_eval.tasks.dna_suite.hallucination import hallucination  # noqa: F401
from dna_eval.tasks.dna_suite.performance import performance  # noqa: F401
from dna_eval.tasks.dna_suite.persona import persona  # noqa: F401
from dna_eval.tasks.dna_suite.style import style  # noqa: F401
from dna_eval.tasks.dna_suite.uncensorship import uncensorship  # noqa: F401
