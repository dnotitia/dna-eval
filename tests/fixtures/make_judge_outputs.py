"""Regenerate ``judge_outputs.json``: what DeepEval's GEval makes of raw judge replies.

Needs ``deepeval==4.0.3`` and ``pyyaml`` (not dependencies of this package); run from the
repository root::

    python tests/fixtures/make_judge_outputs.py

A GEval metric is built from ``src/dna_eval/rubrics/performance.yaml``, and each reply below
is fed to ``GEval.a_measure`` with the judge's HTTP client replaced by a stub that returns the
reply verbatim, so DeepEval's own code path runs unchanged: for a model name it does not know,
GEval skips logprobs, asks for a ``ReasonScore`` schema, gets plain text back,
``trim_and_load_json`` + ``model_validate`` it, and normalises the score to 0-1. The fixture
records the normalised score and the reason, or ``"error"`` when DeepEval raises.

``cases`` are reply texts (the message ``content``); ``transport`` are whole message shapes a
judge server can send, e.g. with a reasoning parser on (``reasoning_content``).
"""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

os.environ.setdefault("OPENAI_API_KEY", "stub")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import GEval  # noqa: E402
from deepeval.metrics.g_eval import Rubric  # noqa: E402
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # noqa: E402

OUT = Path(__file__).with_name("judge_outputs.json")
RUBRIC = Path(__file__).resolve().parents[2] / "src" / "dna_eval" / "rubrics" / "performance.yaml"


def build_metric() -> GEval:
    r = yaml.safe_load(RUBRIC.read_text(encoding="utf-8"))
    return GEval(
        name=r["name"],
        evaluation_steps=r["steps"],
        evaluation_params=[LLMTestCaseParams(p) for p in r["params"]],
        rubric=[Rubric(score_range=tuple(b["range"]), expected_outcome=b["outcome"]) for b in r["rubric"]],
        threshold=r["threshold"],
        model="judge",   # any name DeepEval has no capability data for
    )

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

TRANSPORT = {
    "json_in_reasoning_content": {"content": None, "reasoning_content": '{"score": 8, "reason": "ok"}'},
    "reasoning_content_and_json": {"content": '{"score": 7, "reason": "ok"}', "reasoning_content": "생각 {메모}"},
    "json_with_think_block": {"content": '<think>\n점수 {7}\n</think>\n\n{"score": 7, "reason": "ok"}'},
    "json_after_think_no_braces": {"content": '<think>\n7점이 적당함\n</think>\n\n{"score": 7, "reason": "ok"}'},
    "null_content": {"content": None},
}


def _stub_client(message: dict):
    async def create(**_):
        return SimpleNamespace(
            model="stub",
            choices=[SimpleNamespace(message=SimpleNamespace(**message))],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def judge(metric, message: dict) -> dict:
    metric.model.load_model = lambda *a, **k: _stub_client(message)
    tc = LLMTestCase(input="q", actual_output="a", expected_output="e")
    try:
        await metric.a_measure(tc, _show_indicator=False, _log_metric_to_confident=False)
    except Exception as e:  # noqa: BLE001  (any raise is a failed judge call)
        return {"error": type(e).__name__}
    return {"score": metric.score, "reason": metric.reason}


async def main() -> None:
    metric = build_metric()
    cases = [{"id": k, "raw": v, "expected": await judge(metric, {"content": v})} for k, v in RAW.items()]
    transport = [{"id": k, "message": v, "expected": await judge(metric, {"reasoning_content": None, **v})}
                 for k, v in TRANSPORT.items()]
    doc = {"source": "deepeval 4.0.3 GEval built from rubrics/performance.yaml (score range 0-10)",
           "cases": cases, "transport": transport}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for c in cases + transport:
        print(f"{c['id']:28} {c['expected']}")


if __name__ == "__main__":
    asyncio.run(main())
