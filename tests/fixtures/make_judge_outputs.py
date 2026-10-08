"""Regenerate ``judge_outputs.json``: what DeepEval's GEval makes of raw judge replies.

Run with the llm-model-test venv (it has ``deepeval``), from its checkout::

    cd llm-model-test && .venv/bin/python <dna-eval>/tests/fixtures/make_judge_outputs.py

Each raw reply below is fed to ``metrics.ModelPerformanceMetric.a_measure`` with the judge's
HTTP client replaced by a stub that returns the reply verbatim, so the whole DeepEval path
runs unchanged: for a model DeepEval does not know (gemma), GEval skips logprobs, asks for a
``ReasonScore`` schema, gets plain text back, ``trim_and_load_json`` + ``model_validate`` it,
and normalises the score to 0-1. The fixture records the normalised score and the reason, or
``"error"`` when DeepEval raises (llm-model-test then retries the judge, then scores 0).
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "stub")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
sys.path.insert(0, os.getcwd())

from deepeval.test_case import LLMTestCase  # noqa: E402

import metrics  # noqa: E402  (llm-model-test/metrics.py)

OUT = Path(__file__).with_name("judge_outputs.json")

RAW = {
    "plain": '{"score": 8, "reason": "요구사항을 대부분 충족함"}',
    "reason_first": '{"reason": "ok", "score": 9}',
    "code_fence": '```json\n{"score": 7, "reason": "ok"}\n```',
    "prose_around": 'Here is my evaluation:\n{"score": 6, "reason": "partial"}\nThanks.',
    "float_score": '{"score": 7.5, "reason": "ok"}',
    "float_score_high": '{"score": 8.9, "reason": "ok"}',
    "string_score": '{"score": "8", "reason": "ok"}',
    "string_float_score": '{"score": "7.5", "reason": "ok"}',
    "trailing_comma": '{"score": 8, "reason": "ok",}',
    "missing_close_brace": '{"score": 8, "reason": "ok"',
    "braces_in_reason": '{"score": 5, "reason": "형식 {a: 1}을 지키지 않음"}',
    "nested_object": '{"score": 4, "reason": "ok", "detail": {"a": 1}}',
    "extra_keys": '{"score": 3, "reason": "ok", "confidence": 0.9}',
    "out_of_range": '{"score": 12, "reason": "ok"}',
    "negative": '{"score": -1, "reason": "ok"}',
    "zero": '{"score": 0, "reason": "완전히 틀림"}',
    "think_prefix": '<think>score는 {7}점 정도</think>\n{"score": 7, "reason": "ok"}',
    "two_objects": '{"score": 2, "reason": "draft"}\n{"score": 9, "reason": "final"}',
    "missing_reason": '{"score": 8}',
    "missing_score": '{"reason": "ok"}',
    "null_score": '{"score": null, "reason": "ok"}',
    "word_score": '{"score": "eight", "reason": "ok"}',
    "fraction_score": '{"score": "8/10", "reason": "ok"}',
    "bool_score": '{"score": true, "reason": "ok"}',
    "numeric_reason": '{"score": 8, "reason": 123}',
    "null_reason": '{"score": 8, "reason": null}',
    "list_reason": '{"score": 8, "reason": ["a", "b"]}',
    "single_quotes": "{'score': 8, 'reason': 'ok'}",
    "no_json": "점수는 8점입니다.",
    "empty": "",
    "array_only": "[8]",
}


def _stub_client(reply: str):
    async def create(**_):
        return SimpleNamespace(
            model="stub",
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def judge(metric, reply: str) -> dict:
    metric.model.load_model = lambda *a, **k: _stub_client(reply)
    tc = LLMTestCase(input="q", actual_output="a", expected_output="e")
    try:
        await metric.a_measure(tc, _show_indicator=False, _log_metric_to_confident=False)
    except Exception as e:  # noqa: BLE001  (any raise = judge failure for llm-model-test)
        return {"error": type(e).__name__}
    return {"score": metric.score, "reason": metric.reason}


async def main() -> None:
    metric = metrics.ModelPerformanceMetric
    cases = [{"id": k, "raw": v, "expected": await judge(metric, v)} for k, v in RAW.items()]
    doc = {"source": "deepeval GEval (llm-model-test metrics.ModelPerformanceMetric), score_range 0-10",
           "cases": cases}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for c in cases:
        print(f"{c['id']:22} {c['expected']}")


if __name__ == "__main__":
    asyncio.run(main())
