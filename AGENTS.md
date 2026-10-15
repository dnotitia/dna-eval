# AGENTS.md

Entry point for AI coding agents. Read `README.md` first; `docs/methodology.md` is how the suite
computes its scores.

## Rules

- **Scores must not move by accident.** The judge prompt (`src/dna_eval/rubric_judging.py`),
  the rubrics (`src/dna_eval/rubrics/`), the generation settings and the pinned dataset revisions
  decide the numbers. Change them only when asked; then raise `SUITE_VERSION` in
  `src/dna_eval/tasks/__init__.py` and add a row to the version table in `docs/methodology.md`.
- **Parity tests stay green.** `tests/parity/` checks that judging reproduces DeepEval GEval. A
  failing parity test means a behaviour changed; do not edit its expected values to pass.
- **Fixtures are generated.** Regenerate files under `tests/fixtures/` with their scripts.
- **Real endpoints need permission.** Tests use `mockllm` and `tests/parity/fake_openai.py`.
  Ask before anything that calls a real model server or a GPU.
- **This repository is public.** Describe behaviour; keep internal names, hosts and paths out.
- **Commits are the maintainer's.** Leave committing and pushing to them unless asked.

## Commands

```bash
python -m pytest                      # must pass before you report work as done
python -m pytest -m "not network"     # without Hub downloads
inspect list tasks
```

For an Inspect API you are unsure of: https://inspect.aisi.org.uk/llms.txt
