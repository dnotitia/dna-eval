# 0002. Folders by Inspect type, one implementation per file

- Status: accepted
- Date: 2026-10-12

## Context

Code had started to collect in shared modules and helper folders, mixing unrelated roles.
Contributors need to find a scorer or solver by where Inspect would look for it.

## Decision

- Registered Inspect objects live in folders named after their type: `solvers/`, `scorers/`,
  `metrics/`, `tasks/`. One implementation per file; the file holds that object and its own
  settings (defaults, placeholders), nothing else.
- A file may register one implementation under several names when the names come from data
  (`scorers/rubric_scorer.py` registers one scorer per rubric file).
- Domain logic lives in modules named for what they hold: `rubrics/`, `prompts/`, `rubric_judging.py`,
  `dataset.py`.
- `utils/` holds only helpers with no suite logic, named for what they operate on (`json`,
  `stats`, `model_output`, `retry`).
- A helper used by one caller stays with that caller. Being used in two places is not, by
  itself, a reason to move code; there is no catch-all "common" module.

## Consequences

More, smaller files; each answers one question. New code has an obvious home.

## Alternatives considered

- Folders per evaluation method (judge-based, match-based, ...): mixes object types in one folder.
- A "common" folder for anything shared: becomes a dumping ground.
