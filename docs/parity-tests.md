# DeepEval parity tests

The suite replaces a DeepEval-based harness (`llm-model-test`). Numbers published from the
old harness must remain comparable, so every task carries three parity checks against it.
They live in `tests/parity/` and are parameterised over `tests/parity/cases.py`.

| # | What must match DeepEval | Test | Fixture |
| --- | --- | --- | --- |
| T1 | The request to the model under evaluation: one user turn holding the sample input, no system prompt, `max_tokens` 65536, thinking via `chat_template_kwargs`; one judge call per rubric at temperature 0 | `test_generation.py` (runs the task on `mockllm`) | — |
| T2 | The prompt the judge receives, byte for byte | `test_judge_prompt.py` | `fixtures/judge_prompts/<rubric>.txt` |
| T3 | Per-run 15% trimmed mean and pass count | `test_aggregation.py` | `fixtures/deepeval_runs/<task>.json` |

T1 needs read access to the `dnotitia/dna_*` datasets (marker `network`); T2 and T3 are offline.

## Adding a task

1. **Judge prompt fixture (T2)** — render the rubric with DeepEval itself, from a checkout
   of `llm-model-test` (its venv has `deepeval`):

   ```python
   from deepeval.metrics.g_eval.template import GEvalTemplate
   from deepeval.metrics.g_eval.utils import (construct_test_case_string,
                                              construct_g_eval_params_string, format_rubrics)
   from deepeval.test_case import LLMTestCase
   import metrics                                    # llm-model-test/metrics.py

   m = metrics.<TheMetric>
   tc = LLMTestCase(input="...", actual_output="...", expected_output="...")   # pick any sample
   steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(m.evaluation_steps))
   open("<rubric>.txt", "w", encoding="utf-8").write(GEvalTemplate.generate_evaluation_results(
       evaluation_steps=steps,
       test_case_content=construct_test_case_string(m.evaluation_params, tc),
       parameters=construct_g_eval_params_string(m.evaluation_params),
       rubric=format_rubrics(m.rubric), score_range=m.score_range))
   ```

   Save it as `tests/fixtures/judge_prompts/<rubric>.txt` and record the same
   `input` / `actual_output` / `expected_output` in `cases.py` as `judge_input`.

2. **Per-run fixture (T3)** — take the last 15 rows for the model from `score.log`
   (`["<label>", <mean>, <passed>]`) into `tests/fixtures/deepeval_runs/<task>.json`:

   ```json
   {"runs": [{"mean": 0.90, "passed": 27}, ...], "expected": {"trimmed_mean": 0.8982, "trimmed_passed": 26.91}}
   ```

   `expected` is DeepEval's `finalize_with_trimmed_mean` applied to `runs` (15% trim,
   `int(n*p)` dropped from each end).

3. **Register** the task in `cases.py` with its rubrics, `judge_input`, and the number of
   judge calls per sample (1 for a single rubric, N for a composite).

4. `pytest tests/parity` — all three tests pick the new case up automatically.
