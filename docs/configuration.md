# Configuration

What a run reads from the environment, and where models, judges and outputs are set. Most of
it belongs to Inspect, `huggingface_hub` or NLTK; dna-eval itself reads one variable.

## Environment variables

| variable | read by | used for | default |
| --- | --- | --- | --- |
| `DNA_EVAL_CACHE_DIR` | dna-eval | files dna-eval keeps between runs (see [Cache](#cache)) | `~/.cache/dna_eval` |
| `HF_TOKEN` | `huggingface_hub` | reading the `dnotitia/dna_*` datasets and gated benchmarks (KMMLU-Pro, HLE); the account must have accepted each gated dataset's terms | cached `huggingface-cli login` |
| `HF_HOME` | `huggingface_hub`, `datasets` | where Hub downloads are cached | `~/.cache/huggingface` |
| `VLLM_BASE_URL`, `VLLM_API_KEY` | Inspect, `vllm/<model>` models | the server of the model under evaluation | unset: Inspect starts a local vLLM server on this machine |
| `<SERVICE>_BASE_URL`, `<SERVICE>_API_KEY` | Inspect, `openai-api/<service>/` models | any OpenAI-compatible server; `<SERVICE>` is the service name in upper case (`openai-api/judge/...` reads `JUDGE_BASE_URL`, `JUDGE_API_KEY`) | none: required |
| `NLTK_DATA` | NLTK | where the IFEval / IFBench verifiers look for, and download, NLTK data | `~/nltk_data` |
| `INSPECT_LOG_DIR` | Inspect | where logs are written (same as `--log-dir`) | `./logs` |

Inspect reads many more variables (one pair per provider, `INSPECT_EVAL_*` defaults for CLI
options); see its [options reference](https://inspect.aisi.org.uk/options.html).

### `.env` files

Inspect loads the first `.env` file found from the working directory upward, before a run. A
variable already set in the shell wins over the file, except inside VS Code, where the file
wins. A relative `INSPECT_LOG_DIR` in the file is resolved against the file's directory. Keep
`.env` out of version control: it holds endpoints and keys.

The tests never read models from these variables (decision record 0013), so a `.env` in the
repository cannot send test traffic to a real server.

## Models and roles

The model under evaluation is `--model`. Other models are bound to named roles; a task that
needs a role fails without it, rather than falling back to the evaluated model (decision
record 0010).

| role | needed by | binds |
| --- | --- | --- |
| `grader` | every DNA suite task, `hle`, `imo_answerbench` | the judge |
| `reference` | `style`, when references are generated instead of read from the Hub | the model whose answers the style judge compares against |

```bash
inspect eval dna_eval/performance --model vllm/<model> \
  --model-role grader=openai-api/judge/<judge-model>
```

## Run options that are often changed

| option | effect |
| --- | --- |
| `-T name=value` | a task argument: `thinking`, `epochs`, a dataset `revision`, ... (each task's docstring lists them) |
| `--max-connections N` | concurrent requests per model |
| `--attempt-timeout S` | time limit for one request; a request that runs out is retried |
| `--timeout S` | time limit for a call including all its retries |
| `--log-dir DIR` | where logs are written |
| `--limit N` | run the first N samples only |

## Cache

`DNA_EVAL_CACHE_DIR` holds:

```
style/<dataset revision>/<reference>.<settings hash>.parquet   generated style references (+ .json provenance)
benchmarks/gpqa_diamond/<sha256>.csv                           the GPQA-Diamond question file
benchmarks/imo_answerbench/answerbench_v2.<commit>.csv         the IMO-AnswerBench CSV
```

Downloaded files are checked against their pinned SHA-256 on every read, so a damaged cache
file is an error, not a silent change. Deleting the directory is safe: files are downloaded
or generated again on the next run. Hub datasets are cached by `huggingface_hub` instead
(`HF_HOME`).
