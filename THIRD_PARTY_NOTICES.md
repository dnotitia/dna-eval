# Third-party notices

dna-eval is licensed under the Apache License 2.0 (`LICENSE`). It includes the third-party
material below, each under its own licence. Benchmark datasets are not included: tasks
download them at run time, under the licences listed in `docs/benchmarks.md`.

Every file below that holds third-party material says so in a comment and points here.

## Not under the Apache License

### KoBALT prompts: CC-BY-NC-4.0 (non-commercial use only)

- Where: `SYSTEM`, `PROMPT` and `ANSWER_PATTERN` in `src/dna_eval/tasks/benchmarks/kobalt.py`, between
  `SPDX-SnippetBegin` and `SPDX-SnippetEnd`
- Source: https://github.com/snunlp/KoBALT, files evaluation_protocol.md and data/inference_gpt.py
- Authors: Hyopil Shin, Sangah Lee, Dongjun Jang, Wooseok Song, Jaeyoon Kim, Chaeyoung Oh,
  Hyemi Jo, Youngchae Ahn, Sihyun Oh, Hyohyeong Chang, Sunkyoung Kim, Jinsik Lee (2025),
  "KoBALT: Korean Benchmark For Advanced Linguistic Tasks"
- Licence: Creative Commons Attribution-NonCommercial 4.0 International,
  https://creativecommons.org/licenses/by-nc/4.0/
- Changes: the user prompt takes a `{question}` placeholder; the strings are otherwise as published.

This snippet, and the `kobalt` task that uses it, may be used for non-commercial purposes only.
The KoBALT dataset has the same licence.

## Apache License 2.0

The licence text is in `LICENSE`.

| source | material | files | changes |
| --- | --- | --- | --- |
| [confident-ai/deepeval](https://github.com/confident-ai/deepeval), Copyright 2024 Confident AI Inc. | the GEval results prompt; `trim_and_load_json` | `src/dna_eval/rubric_judging.py` (`GEVAL_RESULTS`), `src/dna_eval/utils/json.py` | prompt as a Jinja template string rendering the same bytes; function ported to Python unchanged in behaviour |
| [TIGER-AI-Lab/MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | the instruction, question layout and answer patterns of `evaluate_from_api.py` | `src/dna_eval/tasks/benchmarks/mmlu_pro.py` | 0-shot; the last answer pattern and the random fallback are dropped |
| [allenai/IFBench](https://github.com/allenai/IFBench), Copyright Allen Institute for AI; Google Research Authors | the instruction verifiers | `src/dna_eval/vendor/ifbench/` | listed in `src/dna_eval/vendor/ifbench/NOTICE.md` |

## MIT License

| source | material | files |
| --- | --- | --- |
| [centerforaisafety/hle](https://github.com/centerforaisafety/hle), Copyright (c) 2025 centerforaisafety | the system prompt, the judge prompt, `calib_err` | `src/dna_eval/tasks/benchmarks/hle.py`, `src/dna_eval/scorers/hle_judge.py` (`JUDGE_PROMPT`), `src/dna_eval/metrics/calibration_error.py` |
| [openai/simple-evals](https://github.com/openai/simple-evals), Copyright (c) 2024 OpenAI | the GPQA prompt and answer pattern | `src/dna_eval/tasks/benchmarks/gpqa_diamond.py` |
| [eth-sri/matharena](https://github.com/eth-sri/matharena), Copyright (c) 2024 SRI Lab, ETH Zurich | the AIME instruction | `src/dna_eval/tasks/benchmarks/aime.py` |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), Copyright (c) 2020 EleutherAI | the HRM8K and CLIcK prompts; HRM8K answer extraction and normalisation | `src/dna_eval/tasks/benchmarks/hrm8k_ksm.py`, `src/dna_eval/tasks/benchmarks/click.py`, `src/dna_eval/utils/math_answer.py` |
| [hendrycks/math](https://github.com/hendrycks/math), Copyright (c) 2021 Dan Hendrycks | the answer normalisation (through lm-evaluation-harness) | `src/dna_eval/utils/math_answer.py` |

Each of these is distributed under the following terms, with the copyright line given above:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD 3-Clause License

| source | material | files |
| --- | --- | --- |
| [LG-AI-EXAONE/KMMLU-Pro](https://github.com/LG-AI-EXAONE/KMMLU-Pro), Copyright (c) 2025 the LG AI Research | the prompt and answer pattern; the licence exam structure and pass rules of `print_score.py` | `src/dna_eval/tasks/benchmarks/kmmlu_pro.py`, `src/dna_eval/tasks/benchmarks/click.py` (answer pattern), `src/dna_eval/metrics/license_pass.py` |

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Creative Commons Attribution 4.0

| source | material | files | changes |
| --- | --- | --- | --- |
| Luong et al. (2025), "Towards Robust Mathematical Reasoning", arXiv 2511.01846, https://arxiv.org/abs/2511.01846 | the AnswerAutoGrader prompt (Appendix A.5) | `src/dna_eval/scorers/imo_answer_grader.py` (`GRADER_PROMPT`) | line wraps joined; typeset quotes written as backticks; section "# 2. Extraction Protocol" added by this project |

Licence: https://creativecommons.org/licenses/by/4.0/
