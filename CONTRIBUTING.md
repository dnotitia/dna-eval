# Contributing to dna-eval

These are the project's working rules, for people and AI agents alike. Read this before
changing anything; `docs/architecture.md` explains how the code fits together, and
`docs/decisions/` records why.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export HF_TOKEN=...                 # read access to the dnotitia/dna_* datasets and gated benchmarks
python -m pytest                    # whole suite, about a minute, no GPU
python -m pytest -m "not network"   # skip the tests that download data
inspect list tasks                  # the registered tasks
```

Inspect reads a `.env` file from the working directory upward (keep it out of git; it is
ignored); `docs/configuration.md` lists the variables a run reads. Run `python -m pytest` rather than a bare `pytest` so the tests use this environment.

## Principles

Each one is a decision record; follow the link for the reasoning and the rejected alternatives.

- **Inspect-native.** Tasks, solvers, scorers and metrics are Inspect objects, and Inspect's own
  mechanisms (model roles, the registry, logs, headline metrics) come first. Add our own only
  where Inspect falls short, and record why. (0001)
- **Rubrics are data.** A new judge metric is a YAML file in `src/dna_eval/rubrics/`, not code.
  Where a judge field comes from is declared in the rubric's `sources`. (0003)
- **Folders by Inspect type.** Registered objects go in `solvers/`, `scorers/`, `metrics/`,
  `tasks/`, one implementation per file. Domain logic goes in a module named for it;
  `utils/` holds only helpers with no suite logic. A helper stays with its single caller. (0002)
- **Dependencies point down.** `tasks` → `solvers`/`scorers`/`metrics` → `rubric_judging`/`prompts`/
  `rubrics`/`dataset` → `utils`. Solvers and scorers exchange data through keys a rubric names,
  never through imports. (0011)
- **Pinned and explicit.** Datasets are pinned to a commit (0008); model roles are required
  (0010); suite-wide settings live in `src/dna_eval/tasks/dna_suite/__init__.py` (0007), and
  benchmark settings in `src/dna_eval/tasks/benchmarks/__init__.py` (0016).
- **Official protocol for benchmarks.** A standard benchmark follows its authors' code; every
  departure is listed in the task's docstring (0016). Copied checking code stays unchanged in
  `src/dna_eval/vendor/` with its licence (0017).
- **Behaviour, not origin.** Code, docs, names and fixtures describe what a thing does in terms
  a reader of this repository can check, in English. (0014)

## The ported DNA suite

The tasks in `src/dna_eval/tasks/dna_suite/` were ported from a DeepEval GEval harness, and
their numbers must stay comparable with the ones it produced. For these tasks only, a change
that moves a score keeps DeepEval parity, pinned by `tests/parity/`, or is recorded as a
deliberate deviation (0001, 0004–0007). New tasks do not carry this constraint.

## Making a change

- **A new task:** follow `docs/adding-a-task.md`. Register it in `src/dna_eval/_registry.py`.
- **A suite rubric, the judge template, or suite settings:** these move the suite's scores. Say
  so in the commit message; a deliberate change also updates the parity fixtures and, if it
  departs from DeepEval, adds a decision record.
- **Parity fixtures** (`tests/fixtures/`) are generated, never hand-edited: judge prompts and
  replies with DeepEval's own code (`tests/fixtures/make_judge_outputs.py`), runs from recorded
  evaluations.
- **A decision:** if a change sets or reverses a rule others must follow (a new folder, a new
  dependency direction, a deviation from parity), add a record in `docs/decisions/` from
  `0000-template.md`. Accepted records are binding; to change one, supersede it with a new
  record instead of editing it. The only edit an accepted record takes is updating a reference
  to a renamed file or module.
- **Tests:** add or update a test with every behaviour change. Tests never call real model
  endpoints: use `mockllm` or `tests/parity/fake_openai.py`, and bind models by `base_url`, not by
  environment variables (0013). Mark tests that need the network (the Hub, pinned downloads) with `@pytest.mark.network`; a
  test of a gated dataset skips with the reason when the token has no access.

## Done means

1. `python -m pytest` passes.
2. New behaviour has a test; changed behaviour has an updated one.
3. Docs that describe the changed behaviour are updated: `README.md`, `docs/architecture.md`
   (modules, contracts, vocabulary), the task guides, and a decision record if a rule changed.
4. `tests/test_docs.py` passes: every repository path these documents name exists, and the
   decision index matches the records.

## Commits and pull requests

- Branch from `main` and open a pull request against `main`; one topic per branch.
- Commit subject: `<type>: <summary>`, lower case, e.g. `fix: keep the reasoning placeholder`.
  Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- A pull request says what changed, why, how it was tested, and which decision records it
  follows or adds.
