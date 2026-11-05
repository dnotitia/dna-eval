"""Entry point Inspect imports for this package: everything that can be looked up by name.

``inspect eval dna_eval/<task>`` and ``inspect score --scorer dna_eval/<rubric>`` resolve
through the modules imported here. Importing ``scorers.rubric_scorer`` registers one scorer
per rubric in ``rubrics/``, whether or not a task uses it.
"""
from dna_eval.metrics import (answer_diagnostics, call_failures, judge_failures, license_pass,  # noqa: F401
                              macro_accuracy, per_run_trimmed, weighted_accuracy)
from dna_eval.scorers import (answer_judge, checklist_grade, letter_choice, math_answer,  # noqa: F401
                              rubric_scorer, simpleqa_grade)
from dna_eval.solvers import dna_generate, style_reference  # noqa: F401
from dna_eval.tasks.benchmarks.click import click  # noqa: F401
from dna_eval.tasks.benchmarks.haerae_bench import haerae_bench  # noqa: F401
from dna_eval.tasks.benchmarks.haerae_vision import haerae_vision  # noqa: F401
from dna_eval.tasks.benchmarks.hrm8k_ksm import hrm8k_ksm  # noqa: F401
from dna_eval.tasks.benchmarks.kbl import kbl  # noqa: F401
from dna_eval.tasks.benchmarks.kmmlu_hard import kmmlu_hard  # noqa: F401
from dna_eval.tasks.benchmarks.kmmlu_pro import kmmlu_pro  # noqa: F401
from dna_eval.tasks.benchmarks.kmmlu_redux import kmmlu_redux  # noqa: F401
from dna_eval.tasks.benchmarks.kmmmu import kmmmu  # noqa: F401
from dna_eval.tasks.benchmarks.kobalt import kobalt  # noqa: F401
from dna_eval.tasks.benchmarks.kormedmcqa import kormedmcqa  # noqa: F401
from dna_eval.tasks.benchmarks.kosimpleqa import kosimpleqa  # noqa: F401
from dna_eval.tasks.degeneration import degeneration  # noqa: F401
from dna_eval.tasks.hallucination import hallucination  # noqa: F401
from dna_eval.tasks.performance import performance  # noqa: F401
from dna_eval.tasks.persona import persona  # noqa: F401
from dna_eval.tasks.style import style  # noqa: F401
