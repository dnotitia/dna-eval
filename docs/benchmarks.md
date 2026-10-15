# Standard benchmarks

Tasks in `src/dna_eval/tasks/benchmarks/` reproduce published benchmarks so that a model's
numbers can be set next to other model reports. Each follows the benchmark's official
protocol, adapted only where reasoning models need it (decision record 0016). The task's
docstring is the full account: the source it follows, and every departure from it.

## Running

```bash
inspect eval dna_eval/gpqa_diamond --model vllm/<model>
inspect eval dna_eval/hle --model vllm/<model> --model-role grader=openai-api/judge/<judge-model>
inspect eval dna_eval/ifeval --model vllm/<model> -T thinking=false -T epochs=4
```

Every task takes `thinking` (default `true`, sent as `chat_template_kwargs.enable_thinking`),
`epochs` (runs averaged into pass@1), and the pinned data version (`revision`, `commit` or
`sha256`). No task sets `max_tokens`: serve the model with the context length it is meant to
use, and read `truncated` in the results to see how many answers ran out of room.

## Tasks

| task | data (size) | answer read from the final content | headline | runs | judge |
| --- | --- | --- | --- | --- | --- |
| `gpqa_diamond` | simple-evals CSV (198) | last `Answer: X`, options shuffled per run | `accuracy` | 8 | — |
| `mmlu_pro` | `TIGER-Lab/MMLU-Pro` test (12,032) | last `answer is (X)`, then `Answer: X`; 0-shot | `accuracy` | 1 | — |
| `kmmlu_pro` | `LGAI-EXAONE/KMMLU-Pro` (2,822, gated) | last `정답: X` | `weighted_accuracy` | 1 | — |
| `click` | `EunsuKim/CLIcK` (1,995) | last `정답: X`; all options listed | `accuracy` | 1 | — |
| `kobalt` | `snunlp/KoBALT-700` (700) | `정답은 X입니다`, exactly one distinct letter | `accuracy` | 1 | — |
| `aime2025`, `aime2026` | `MathArena/aime_*` (30 each) | last `\boxed{}`, else last integer | `accuracy` | 8 | — |
| `hrm8k_ksm` | `HAERAE-HUB/HRM8K` KSM (1,428) | last `\boxed{}`; string or math-verify equivalence | `accuracy` | 1 | — |
| `imo_answerbench` | `google-deepmind/superhuman` CSV (400) | the grader reads the answer | `accuracy` | 8 | `grader` |
| `hle` | `cais/hle` text-only (2,158, gated) | the judge reads `Answer:` | `accuracy` | 1 | `grader` |
| `ifeval` | `google/IFEval` (541) | the whole response | `prompt_level_strict` | 1 | — |
| `ifbench` | `allenai/IFBench_test` (300) | the whole response | `prompt_level_loose` | 8 | — |
| `scicode` | `SciCode1/SciCode` test (65 problems, 288 sub-steps) | each sub-step's code, run against its tests in a sandbox | `accuracy` (problems) | 3 | — |

Gated datasets need an `HF_TOKEN` whose account has accepted the dataset's terms. `scicode` runs
each sub-step's tests in a Docker-compatible sandbox (`--sandbox` picks the provider), downloads a
1 GB test-data file once, and needs `pip install "dna-eval[scicode]"`; `hrm8k_ksm` needs
`pip install "dna-eval[math]"` (math-verify). Agent tasks are in
[agentic.md](agentic.md).

## Data licences

dna-eval does not include benchmark data; each task downloads it when it runs. Using it is
governed by the dataset's own licence and, for gated datasets, the terms accepted on the Hub.
Check them before using a task for anything but non-commercial research. A new task adds its
row here (`python scripts/license_report.py` looks up Hub licences).

| task | dataset licence | conditions to note |
| --- | --- | --- |
| `gpqa_diamond` | CC-BY-4.0 | do not reveal examples online in plain text or images (the dataset's terms) |
| `mmlu_pro` | MIT | |
| `kmmlu_pro` | CC-BY-NC-ND-4.0 | **non-commercial use only** (also agreed when requesting access) |
| `click` | none stated | no licence is published for the dataset |
| `kobalt` | CC-BY-NC-4.0 | **non-commercial use only**; the task's prompts carry the same licence (`THIRD_PARTY_NOTICES.md`) |
| `aime2025`, `aime2026` | CC-BY-NC-SA-4.0 | **non-commercial use only** |
| `hrm8k_ksm` | MIT | |
| `imo_answerbench` | CC-BY-4.0 | |
| `hle` | MIT | keep questions out of training data (the dataset carries a canary string) |
| `ifeval` | Apache-2.0 | |
| `ifbench` | ODC-BY-1.0 | |
| `scicode` | Apache-2.0 | its 1 GB test-data file comes from a third-party Hub mirror of the authors' file (same SHA-256) |
| `tau_bench` | MIT | data from the benchmark's GitHub repository at a pinned commit |
| `automationbench` | MIT | the public split; the official leaderboard uses a private one |

Evaluation logs hold every question and answer. Treat them as copies of the data: in
particular, do not publish logs of `gpqa_diamond` or `hle`.

## What every result reports

- **The headline**, plus `stderr` and the benchmark's own breakdowns (per category, per
  licence for KMMLU-Pro, calibration error for HLE, the four IFEval numbers).
- **`answer_diagnostics`**: the share of answers that were `truncated` or had no readable
  answer (`format_failure`), and `accuracy_excl_truncated`. Both kinds of failure score 0 in the
  headline; these numbers show how much of a gap they explain. The IFEval tasks report
  `truncated` and the `_excl_truncated` variants in their own metric.

## Judges

`hle` and `imo_answerbench` are graded by a model, as their protocols are, through the
`grader` role (required). The official judges (o3-mini for HLE, a Gemini model for
IMO-AnswerBench) are not reproduced by binding another model: a number is comparable only
with runs graded by the same judge, which the log records.

## Third-party code

The IFEval and IFBench verifiers are vendored from `allenai/IFBench` under Apache-2.0
(`src/dna_eval/vendor/ifbench/`, with its `LICENSE` and a `NOTICE.md` listing the local
changes; decision record 0021). They download NLTK data (`punkt_tab`, `stopwords`,
`averaged_perceptron_tagger_eng`) the first time a response is scored.
