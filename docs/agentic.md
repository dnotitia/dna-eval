# Agent tasks

Tasks in `src/dna_eval/tasks/agentic/` run the model under evaluation in a loop: it calls tools,
and in some tasks talks with a simulated user, until it finishes or a limit stops it. They
follow decision record 0020; each task's docstring names the official code it uses and every
departure from it.

| task | benchmark | loop | roles | runs |
| --- | --- | --- | --- | --- |
| `tau_bench` | tau-bench (τ³, v1.0.1): `domain` airline (50), retail (114), telecom (114), banking_knowledge (97) | the official conversation: agent, simulated user and the domain's tools | `user` (simulated user), `grader` (retail and banking judge assertions) | 4 (pass^1 to pass^4) |
| `automationbench` | AutomationBench v1.0.6, public split: `domain` sales, marketing, operations, support, finance, hr (100 each) | tool calls against simulated business apps, up to 50 turns | — | 1 |
| `terminal_bench` | Terminal-Bench `version` 2.1 (89, default) or 4.0 (66, experimental) | an agent `scaffold` in the task's container | — | 5 |
| `swe_bench` | SWE-bench `variant` pro (731, default), pro_hardened (731), verified (500) | an agent `scaffold` in the repository's container, up to 250 steps | — | 1 |
| `deep_swe` | DeepSWE v1.1 (113) | an agent `scaffold` in the repository's container | — | 1 |

## Installing

The benchmarks' own code is a dependency, installed separately (neither is on PyPI):

```bash
# tau_bench (Python 3.12 or 3.13); banking_knowledge with BM25 retrieval also needs rank-bm25
pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench@fc0055dc4e0a316c3f83133267fbd6faaa770992" "audioop-lts; python_version >= '3.13'"
# automationbench (Python 3.13)
pip install "automation-bench @ git+https://github.com/zapier/AutomationBench@4a8e1061254004d9dac807054eed33fad7d1ff14" "verifiers==0.2.1"
```

```bash
# terminal_bench, swe_bench, deep_swe (inspect-harbor needs Python 3.12 or later)
pip install "dna-eval[agentic]"
```

A task whose package is missing stops with this command in its error message. `tau_bench`
downloads its data (about 100 MB) once into the cache directory (`docs/configuration.md`), or
uses `TAU2_DATA_DIR` if set.

## Roles and comparability

A simulated user decides much of a tau-bench score: bind `--model-role user=<model>`
(required). The official runs use GPT-4.1 as the user and as the judge of retail's assertions
(`--model-role grader=<model>`); numbers are comparable only with runs that used the same models.
AutomationBench's leaderboard uses a private split, so public-split numbers are directional.

## Sandboxed tasks

`terminal_bench`, `swe_bench` and `deep_swe` load Harbor-format datasets through
`inspect-harbor`, pinned by digest, and run each task in its own container: the task's image,
resources and network setting. The sandbox provider is yours to choose: `docker` by default,
or any Docker-compatible provider (`--sandbox k8s`, cloud providers); a provider that cannot
build images needs the tasks' images available in a registry it can pull from.

- **Scaffold.** The agent loop is a task argument (`-T scaffold=`): `mini_swe_agent` (default;
  mini-swe-agent 2.2.3, bash only), `react` (Inspect's ReAct agent with bash and python tools),
  `claude_code` or `codex_cli`. Scores depend strongly on it; it is recorded in the log.
  mini-swe-agent is installed inside the container, which needs network access or Python 3.11.
- **Limits.** `-T max_steps=` caps model turns; each task's own agent time limit ends a run
  that passes it. Either is scored on the final state and counted by `agent_diagnostics`.
- **Errors.** A container or verifier that breaks makes the sample an error. Run with
  `--no-fail-on-error --retry-on-error 3`.
- **Terminal-Bench 4.0 is experimental.** Its tasks verify in a separate container, which
  `inspect-harbor` 1.0.0 does not support yet: tests run in the agent's container, and some
  tasks need a GPU. Use 2.1 (the default) for comparable numbers.

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
