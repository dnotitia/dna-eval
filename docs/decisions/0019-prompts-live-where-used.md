# 0019. Prompts live in the module that uses them

- Status: accepted
- Date: 2026-10-14
- Scope: every prompt text: instructions to the model under evaluation and judge prompts.

## Context

Judge prompts were Jinja files in a separate prompts package while instructions to the model under
evaluation were strings in task files, so where a prompt lived depended on its length and
form rather than on any rule. A prompt is coupled to the code that reads the reply: a task's
instruction to its answer pattern, a judge prompt to its verdict parsing.

## Decision

- A prompt is a constant in the module that uses it: a task file, a scorer, or
  `rubric_judging.py` for the rubric judge.
- A prompt several modules use lives in the module they all depend on (as the GEval prompt
  lives in `rubric_judging.py` for every rubric scorer, and the AIME instruction in
  `tasks/benchmarks/aime.py` for both years), never in one of the users.
- A prompt that needs template logic is a Jinja string rendered by `utils/templates.py`.
- The prompts package is removed. Records 0002, 0004 and 0011 only have their references
  to it updated.

## Consequences

A task or scorer reads top to bottom with its prompt in view. Long prompts make their module
longer; byte-for-byte tests keep them exact. Third-party text is marked where it sits (0018).

## Alternatives considered

- All prompts as Jinja files in one folder, flat or split by role or source: separates each
  prompt from the code that parses the reply, and adds a file per one-line instruction.
- Keeping that folder for judge prompts only: the boundary has to be explained by length.
