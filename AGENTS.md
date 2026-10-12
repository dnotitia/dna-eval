# AGENTS.md

Entry point for AI coding agents. People start at `README.md`; the documents below are the
same ones they read.

**Read `CONTRIBUTING.md` before changing anything.** It holds the project's rules, principles
and definition of done; this file only adds what an agent needs on top.

## Read before you act

- **Changing structure** (a new module, moving code, a new import between packages):
  `docs/architecture.md`, then the decision records it cites in `docs/decisions/`.
- **Adding or changing a task:** `docs/adding-a-task.md`; for the style task,
  `docs/running-style-task.md`; for a standard benchmark, `docs/benchmarks.md` and record 0015.
- **Touching `src/dna_eval/vendor/`:** record 0016; checking logic stays as upstream wrote it.
- **Changing a DNA suite rubric, the judge template, or suite settings:** the matching records
  in `docs/decisions/` (0003–0007) first; these change the suite's scores and its DeepEval
  parity. New tasks outside the ported suite need no DeepEval parity (0001).
- **Using an Inspect API you are unsure of:** https://inspect.aisi.org.uk/llms.txt

## Commands

```bash
python -m pytest                      # must pass before you report work as done
python -m pytest -m "not network"     # without Hub downloads
inspect list tasks
```

## Working rules

- **Decision records are binding.** Follow accepted records in `docs/decisions/`. If a task
  needs something an accepted record rules out, stop and ask; if the change is agreed, write a
  superseding record instead of editing the old one.
- **Fixtures are generated.** Regenerate files under `tests/fixtures/` with their scripts or
  recorded runs; never edit expected values by hand to make a test pass.
- **Real endpoints need permission.** Tests and checks use `mockllm` and
  `tests/parity/fake_openai.py`. Ask before anything that calls a real model server or a GPU.
- **Commits are the maintainer's.** Leave committing and pushing to them unless asked.
- **Report what you verified.** State which commands you ran and their result; say so when a
  step was skipped.
