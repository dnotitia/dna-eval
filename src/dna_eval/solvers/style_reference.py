"""``style_reference``: attach a reference model's reasoning and answer to each sample.

References come from the dataset repository (``references/<model>.parquet``, attached to the
samples when the task loads them) or, for a model with no published references, from the
model bound to the ``reference`` role. Generated references are cached on disk in the same
parquet schema as the published ones, so a cache file can be uploaded as it is:

    <cache dir>/style/<dataset revision>/<reference>.<settings hash>.parquet   (+ .json provenance)

The cache dir is ``$DNA_EVAL_CACHE_DIR`` or ``~/.cache/dna_eval``. The settings hash covers
the reference model and the generation settings, so a different model or config never
reuses another's references.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, get_model, model_roles
from inspect_ai.solver import Generate, Solver, TaskState, solver

from dna_eval.solvers.dna_generate import ATTEMPTS
from dna_eval.utils.cache import cache_root
from dna_eval.utils.model_output import reasoning_text
from dna_eval.utils.retry import call_with_retries

logger = logging.getLogger(__name__)

ROLE = "reference"
COLUMNS = ["id", "input", "reference_reasoning", "reference_content"]

# What the judge is shown when the reference has no reasoning, or no answer.
NO_REFERENCE_REASONING = "[참조 모델 추론 없음]"
NO_REFERENCE_ANSWER = "[참조 모델 응답 없음]"

# ``state.store`` key recording where a sample's reference came from: hub, cache or generated.
ORIGIN_KEY = "reference_origin"

# GenerateConfig fields that do not change what the model returns (left out of the hash).
_RUNTIME_FIELDS = {"max_connections", "adaptive_connections", "max_retries", "timeout",
                   "attempt_timeout", "stream_idle_timeout", "cache", "batch", "fail_on_refusal"}


class ReferenceCache:
    """Generated references for one (dataset revision, reference, model, settings)."""

    def __init__(self, reference: str, dataset_revision: str, config: GenerateConfig, refresh: bool):
        self.reference, self.dataset_revision, self.config, self.refresh = reference, dataset_revision, config, refresh
        self.path: Path | None = None
        self._rows: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._write = asyncio.Lock()

    def open(self, model: Model) -> None:
        """Bind to ``model`` and load existing references (unless ``refresh``). Idempotent."""
        if self.path is not None:
            return
        settings = {"model": str(model), "config": self.config.model_dump(exclude=_RUNTIME_FIELDS, exclude_none=True)}
        key = hashlib.md5(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.settings = settings
        self.path = cache_root() / "style" / self.dataset_revision / f"{self.reference}.{key}.parquet"
        if self.path.exists() and not self.refresh:
            self._rows = {r["id"]: r for r in pd.read_parquet(self.path).to_dict("records")}

    async def get(self, sample_id: str, input_text: str, model: Model, attempts: int) -> tuple[dict, str]:
        """(reference row, origin) for one sample; generated at most once per sample."""
        lock = self._locks.setdefault(sample_id, asyncio.Lock())
        async with lock:
            if sample_id in self._rows:
                return self._rows[sample_id], "cache"
            output = await call_with_retries(lambda: model.generate([ChatMessageUser(content=input_text)]), attempts)
            row = {"id": sample_id, "input": input_text,
                   "reference_reasoning": reasoning_text(output) if output else "",
                   "reference_content": output.completion.strip() if output else ""}
            if row["reference_content"] or row["reference_reasoning"]:   # a failed call is not cached
                self._rows[sample_id] = row
                await self._persist()
            return row, "generated"

    async def _persist(self) -> None:
        async with self._write:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rows = sorted(self._rows.values(), key=lambda r: r["id"])
            tmp = self.path.with_suffix(".parquet.tmp")
            pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp, index=False)
            tmp.replace(self.path)
            self.path.with_suffix(".json").write_text(json.dumps({
                "reference": self.reference, "dataset_revision": self.dataset_revision, **self.settings,
                "rows": len(rows), "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")


@solver
def style_reference(reference: str, dataset_revision: str, config: GenerateConfig,
                    refresh: bool = False, attempts: int = ATTEMPTS) -> Solver:
    """Put ``metadata["reference_reasoning"]`` / ``["reference_content"]`` on every sample.

    A sample that already carries them (published references) is left as it is. Otherwise the
    reference comes from the local cache or is generated once per sample, with the same
    retry and answer/reasoning handling as ``dna_generate`` and with ``config``, by the model
    bound to the ``reference`` role (required). ``refresh`` ignores an existing cache file.
    """
    cache = ReferenceCache(reference, dataset_revision, config, refresh)
    warned = False

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        nonlocal warned
        if "reference_content" in state.metadata:
            if ROLE in model_roles() and not warned:
                warned = True
                logger.warning("published references for %r are used; the %r role is ignored "
                               "(pass -T reference_source=generate to generate instead)", reference, ROLE)
            state.store.set(ORIGIN_KEY, "hub")
            return state
        model = get_model(role=ROLE, required=True, config=config)
        cache.open(model)
        row, origin = await cache.get(str(state.sample_id), state.input_text, model, attempts)
        state.metadata["reference_reasoning"] = row["reference_reasoning"] or NO_REFERENCE_REASONING
        state.metadata["reference_content"] = row["reference_content"] or NO_REFERENCE_ANSWER
        state.store.set(ORIGIN_KEY, origin)
        return state

    return solve
