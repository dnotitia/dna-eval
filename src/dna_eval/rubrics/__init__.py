"""Rubrics: what a judge is asked to assess.

Each ``*.yaml`` in this package is one judge pass — evaluation steps, score bands, the fields
the judge sees, and the pass threshold. The schema lives next to the data so a rubric edit
and its validation rule change together.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

PARAM_LABEL = {
    "input": "Input",
    "actual_output": "Actual Output",
    "expected_output": "Expected Output",
    "context": "Context",
}


class RubricBand(BaseModel):
    range: tuple[int, int]
    outcome: str


class Rubric(BaseModel):
    name: str
    params: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    rubric: list[RubricBand] = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)

    @field_validator("params")
    @classmethod
    def _known_params(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if p not in PARAM_LABEL]
        if bad:
            raise ValueError(f"unknown params {bad}; allowed: {sorted(PARAM_LABEL)}")
        return v

    @property
    def lo(self) -> int:
        return min(b.range[0] for b in self.rubric)

    @property
    def hi(self) -> int:
        return max(b.range[1] for b in self.rubric)


def load_rubric(name_or_path: str) -> Rubric:
    """Load a rubric by suite name (``performance``) or by path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = resources.files(__name__).joinpath(f"{name_or_path}.yaml").read_text(encoding="utf-8")
    return Rubric.model_validate(yaml.safe_load(text))
