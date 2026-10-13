# 0010. Model roles are required, never implied

- Status: accepted
- Date: 2026-10-12

## Context

`get_model(role="grader")` returns the model under evaluation when no model is bound to the
role. A run that forgot `--model-role grader=...` would have the model judge its own answers.

## Decision

Every role lookup uses `required=True`: `grader` in `rubric_judging.py`, `reference` in
`solvers/style_reference.py`. A missing role stops the eval with Inspect's
"Model role '<role>' is required" error.

## Consequences

Misconfigured runs fail at the first judged sample instead of producing plausible numbers.
