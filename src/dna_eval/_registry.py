"""Entry point Inspect imports for this package: everything that can be looked up by name.

``inspect eval dna_eval/<task>`` and ``inspect score --scorer dna_eval/<rubric>`` resolve
through the modules imported here. Importing ``scorers.rubric_scorer`` registers one scorer
per rubric in ``rubrics/``, whether or not a task uses it.
"""
from dna_eval.metrics import per_run_trimmed  # noqa: F401
from dna_eval.scorers import rubric_scorer  # noqa: F401
from dna_eval.solvers import dna_generate, style_reference  # noqa: F401
from dna_eval.tasks.dna_suite.degeneration import degeneration  # noqa: F401
from dna_eval.tasks.dna_suite.hallucination import hallucination  # noqa: F401
from dna_eval.tasks.dna_suite.performance import performance  # noqa: F401
from dna_eval.tasks.dna_suite.persona import persona  # noqa: F401
from dna_eval.tasks.dna_suite.style import style  # noqa: F401
from dna_eval.tasks.dna_suite.uncensorship import uncensorship  # noqa: F401
