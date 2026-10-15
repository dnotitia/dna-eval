# Tasks

Each evaluation run in dna-eval is a task: one `@task` function per file, registered in
`src/dna_eval/_registry.py` and run as `inspect eval dna_eval/<task>`. A task file loads the
data and names a solver, scorer, and metrics. Most of these components live outside
`tasks/` and are shared between tasks. This page describes task placement and file contents.
See the step-by-step walkthrough in
[docs/adding-a-task.md](../../../docs/adding-a-task.md).

## Layout

Folders group tasks that share conventions for calling the model, reporting results, and
classifying failures. The folder's `__init__.py` defines those conventions, and its decision
records document them. Add a task to the folder whose conventions it follows.

| folder | scope | conventions (`__init__.py`) | decision records |
| --- | --- | --- | --- |
| `dna_suite/` | an LLM judge scores each answer against a YAML rubric, and the result is compared with the other suite tasks | no system prompt, one user turn, a fixed token budget; 15 runs; headline `trimmed_mean` (the trimmed mean of per-run means) | 0001, 0005-0007 |
| `benchmarks/` | it reproduces a published benchmark from its authors' protocol, and the model answers without tools or a simulated user | thinking on, no output cap; answers read from the final content, never from the reasoning; headline `accuracy`, reported with `answer_diagnostics` (truncated answers, missing answers) | 0016 |
| `agentic/` | the model is given tools or a simulated user, and their responses come back to it during the run | the benchmark's own environment and scoring code; its official turn and time limits, with a run that hits one scored on its final state; the simulated user and the judge bound to required roles; an infrastructure failure is an error, not a score of 0 | 0020 |

Tasks that fit none of these folders require a new folder, an `__init__.py` describing its
conventions, and a decision record documenting them.

## Task file contents

A task file says what is evaluated and how the parts fit together. It holds:

- the data: loading it, turning each record into a `Sample`, and the pinned revision or checksum
- the prompts the model under evaluation reads (0019)
- the `Task(...)`: solver, scorer, metrics, `epochs` and generation config
- task arguments, only for what changes between runs (`thinking`, `epochs`, `revision`, ...)
- a docstring naming the official code or paper it follows, and every departure from it

It does not hold:

- solver, scorer or metric logic: these go in `solvers/`, `scorers/` and `metrics/`, one per file
- judge prompts or rubrics: they stay with their scorer, or in `rubrics/`
- settings shared by its folder (generation config, headline metric): these are in the folder's `__init__.py`
- model names or endpoints: other models are bound to roles at run time (0010)

Register the task in `_registry.py` and add it to the tables below.

## Task list

The bold metric is the headline, the number Inspect shows first. "runs" is the default
`epochs`. `grader` is the judge role and `user` the simulated-user role, bound with
`--model-role <role>=<model>`.

### `dna_suite/`

Each task loads `dnotitia/dna_<name>` at a pinned revision and scores with `rubric_scorer`
against the listed rubrics. The metrics are `rubric_scorer`'s: **`trimmed_mean`**,
`trimmed_passed` and `runs` (`per_run_trimmed`), `mean`, `call_failures`, `judge_failures`.
Every task requires `grader`.

| task | data | solver | rubrics | runs | requires |
| --- | --- | --- | --- | --- | --- |
| `performance` | `dna_performance` | `dna_generate` | `performance` | 15 | |
| `hallucination` | `dna_hallucination` | `dna_generate` | `hallucination` (metrics also per `category`) | 15 | |
| `persona` | `dna_persona` | `dna_generate` | `persona` (the persona sheet as context) | 15 | |
| `uncensorship` | `dna_uncensorship` | `dna_generate` | `uncensorship` | 15 | |
| `degeneration` | `dna_confusion` | `dna_generate` | `language_confusion`, `repetition` | 15 | |
| `style` | `dna_style` | `style_reference`, `dna_generate` | `reasoning_style`, `content_style` | 15 | `reference` when references are generated ([guide](../../../docs/running-style-task.md)) |

### `benchmarks/`

Tasks here, except `ifeval` and `ifbench`, report `accuracy`, `stderr`, and
`answer_diagnostics`, plus accuracy for each value of the fields under "groups". `ifeval`
and `ifbench` report the benchmark's own four numbers instead, with truncated answers counted.

| task | data | solver | scorer | metrics | groups | runs | requires |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `aime2025`, `aime2026` | `MathArena/aime_<year>` | `generate` | `math_answer("integer")` | **`accuracy`** | | 8 | |
| `gpqa_diamond` | simple-evals CSV (by checksum) | `shuffled_choices`, `generate` | `letter_choice` | **`accuracy`** | | 8 | |
| `hle` | `cais/hle` (gated) | `generate` | `hle_judge` | **`accuracy`**, `calibration_error`, `judge_failures` | `category` | 1 | `grader` |
| `hrm8k_ksm` | `HAERAE-HUB/HRM8K` | `generate` | `math_answer("expression")` | **`accuracy`** | `category` | 1 | `dna-eval[math]` |
| `ifeval` | `google/IFEval` | `generate` | `instruction_following` | **`prompt_level_strict`**, and prompt and instruction level, strict and loose (`instruction_accuracy`) | | 1 | |
| `ifbench` | `allenai/IFBench_test` | `generate` | `instruction_following` | **`prompt_level_loose`**, and the rest as in `ifeval` | | 8 | |
| `imo_answerbench` | `google-deepmind/superhuman` CSV | `generate` | `imo_answer_grader` | **`accuracy`**, `judge_failures` | `category` | 8 | `grader` |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` | `generate` | `letter_choice` | **`accuracy`** | `category` | 1 | |
| `kmmlu_pro` | `LGAI-EXAONE/KMMLU-Pro` (gated) | `generate` | `letter_choice` | **`weighted_accuracy`**, `license_pass` | | 1 | |
| `click` | `EunsuKim/CLIcK` | `generate` | `letter_choice` | **`accuracy`** | `category`, `subcategory` | 1 | |
| `kobalt` | `snunlp/KoBALT-700` | `system_message`, `generate` | `letter_choice` | **`accuracy`** | `class`, `level` | 1 | non-commercial use only |
| `scicode` | `SciCode1/SciCode`, and its test data (by checksum) | `scicode_steps` | `scicode_tests` | **`accuracy`**, `sub_step_accuracy` | | 3 | `dna-eval[scicode]`; a sandbox: `docker` (default), `modal`, `daytona` or `e2b` (build the image), `k8s` (`dna-eval-scicode:1` must be in a registry), or `local` (no isolation; host Python with numpy, scipy, sympy and h5py) |

`generate` and `system_message` are Inspect's own solvers.

Sandbox providers other than `docker` and `local` are separate packages: `k8s` is
`inspect-k8s-sandbox`, and `modal`, `daytona` and `e2b` are `inspect-sandboxes`. Select one with
`inspect eval --sandbox <provider>`.

### `agentic/`

| task | data | solver | scorer | metrics | runs | sandbox | requires |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tau_bench` | tau-bench, `domain` airline, retail, telecom or banking_knowledge | `tau_conversation` | `tau_reward` | **`accuracy`**, `pass_hat_k`, `agent_diagnostics` | 4 | none (runs in the eval process) | `user`; `grader` for retail and banking_knowledge; the `tau2` package ([install](../../../docs/agentic.md#installing)) |
| `automationbench` | AutomationBench public split, `domain` sales, marketing, operations, support, finance or hr | `automationbench_agent` | `automationbench_state` | **`accuracy`** (also per `domain`), `partial_credit`, `agent_diagnostics` | 1 | none (runs in the eval process) | the `automation-bench` package and Python 3.13 ([install](../../../docs/agentic.md#installing)) |
| `terminal_bench` | Terminal-Bench `version` 2.1 or 4.0, Harbor hub (by digest) | `agent_solver` (`-T scaffold=`) | `harbor_reward` | **`accuracy`**, `agent_diagnostics` | 5 | `docker` (default), `k8s`, `daytona`, `e2b` (pre-built images); `modal` for 2.1 only (11 tasks of 4.0 use several containers) | `dna-eval[agentic]` |
| `swe_bench` | SWE-bench `variant` pro, pro_hardened or verified, Harbor hub (by digest) | `agent_solver` (`-T scaffold=`) | `harbor_reward` | **`accuracy`**, `agent_diagnostics` | 1 | `docker` (default), `modal`, `daytona`, `e2b` (each task's image is built from its Dockerfile); `k8s` only with the images built and pushed beforehand | `dna-eval[agentic]` |
| `deep_swe` | DeepSWE v1.1, Harbor hub (by digest; gated) | `agent_solver` (`-T scaffold=`) | `harbor_reward` | **`accuracy`**, `agent_diagnostics` | 1 | `docker` (default), `k8s`, `modal`, `daytona`, `e2b` (pre-built images) | `dna-eval[agentic]` |

For `terminal_bench`, `swe_bench` and `deep_swe`, choose the provider with `-T sandbox=<provider>`.
