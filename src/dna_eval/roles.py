"""Names of the model roles tasks bind with ``--model-role <name>=<model>``.

A role lookup always passes ``required=True`` (decision record 0010): a task that needs a
role fails without it instead of silently using the model under evaluation.
"""

GRADER_ROLE = "grader"          # the judge, for every judge-graded scorer
REFERENCE_ROLE = "reference"    # the style task's reference model, when references are generated
USER_ROLE = "user"              # the simulated user of a conversational agent task (tau-bench)
