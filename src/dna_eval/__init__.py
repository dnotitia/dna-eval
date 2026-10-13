"""DNA evaluation suite for Inspect AI.

A task is the combination of four independently replaceable parts: dataset
(``dna_eval.datasets``), generation (``dna_eval.solvers``), scoring
(``dna_eval.scorers``), and aggregation (``dna_eval.metrics``). Rubrics are data,
not code: see ``dna_eval.rubrics``; the judge prompt lives in ``dna_eval.prompts``.
"""
__version__ = "0.1.0.dev0"
