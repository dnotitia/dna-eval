# Agent tasks

Tasks in `src/dna_eval/tasks/agentic/` run the model under evaluation in a loop: it calls tools,
and in some tasks talks with a simulated user, until it finishes or a limit stops it. They
follow decision record 0020; each task's docstring names the official code it uses and every
departure from it.

| task | benchmark | loop | roles | runs |
| --- | --- | --- | --- | --- |
| `tau_bench` | tau-bench (τ³, v1.0.1): `domain` airline (50), retail (114), telecom (114), banking_knowledge (97) | the official conversation: agent, simulated user and the domain's tools | `user` (simulated user), `grader` (retail and banking judge assertions) | 4 (pass^1 to pass^4) |
| `automationbench` | AutomationBench v1.0.6, public split: `domain` sales, marketing, operations, support, finance, hr (100 each) | tool calls against simulated business apps, up to 50 turns | — | 1 |

## Installing

The benchmarks' own code is a dependency, installed separately (neither is on PyPI):

```bash
# tau_bench (Python 3.12 or 3.13); banking_knowledge with BM25 retrieval also needs rank-bm25
pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench@fc0055dc4e0a316c3f83133267fbd6faaa770992" "audioop-lts; python_version >= '3.13'"
# automationbench (Python 3.13)
pip install "automation-bench @ git+https://github.com/zapier/AutomationBench@4a8e1061254004d9dac807054eed33fad7d1ff14" "verifiers==0.2.1"
```

A task whose package is missing stops with this command in its error message. `tau_bench`
downloads its data (about 100 MB) once into the cache directory (`docs/configuration.md`), or
uses `TAU2_DATA_DIR` if set.

## Roles and comparability

A simulated user decides much of a tau-bench score: bind `--model-role user=<model>`
(required). The official runs use GPT-4.1 as the user and as the judge of retail's assertions
(`--model-role grader=<model>`); numbers are comparable only with runs that used the same models.
AutomationBench's leaderboard uses a private split, so public-split numbers are directional.

## Serving the model

Agent tasks need tool calling. With vLLM, start the server with tool calling and the model's
reasoning parser enabled, for example:

```bash
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser> --reasoning-parser <parser>
```

vLLM reads tool calls from the answer content only, not from the reasoning; check that the
parsers match your model's chat template.

## Notes

- Telecom sample ids contain `[ ] |`, which `--sample-id` reads as glob characters; select with
  `--limit` or an escaped pattern.
- `limit_reached` and `mean_turns` (`agent_diagnostics`) show how many runs a limit ended.
