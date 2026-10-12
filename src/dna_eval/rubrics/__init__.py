"""Rubrics: what a judge is asked to assess.

Each ``*.yaml`` in this package is either

- a judge rubric (``Rubric``): one judge pass — evaluation steps, score bands, the fields the
  judge sees, and the pass threshold; or
- a composite rubric (``CompositeRubric``, has ``compose``): several judge rubrics applied to
  the same answer and folded into one score.

The schema lives next to the data so a rubric edit and its validation rule change together.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PARAM_LABEL = {
    "input": "Input",
    "actual_output": "Actual Output",
    "expected_output": "Expected Output",
    "context": "Context",
}


# Where each judge-visible field comes from, unless a rubric's ``sources`` says otherwise.
#   input / answer / reasoning / target : the sample input, the model's answer, the model's
#   reasoning, the sample target; ``metadata.<key>`` : a field of the sample's metadata.
DEFAULT_SOURCES = {
    "input": "input",
    "actual_output": "answer",
    "expected_output": "target",
    "context": "metadata.context",
}
SOURCE_KINDS = ("input", "answer", "reasoning", "target")


class RubricBand(BaseModel):
    range: tuple[int, int]
    outcome: str


class Rubric(BaseModel):
    name: str
    params: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    rubric: list[RubricBand] = Field(min_length=1)
    threshold: float = Field(ge=0.0, le=1.0)
    sources: dict[str, str] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _known_params(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if p not in PARAM_LABEL]
        if bad:
            raise ValueError(f"unknown params {bad}; allowed: {sorted(PARAM_LABEL)}")
        return v

    @field_validator("sources")
    @classmethod
    def _known_sources(cls, v: dict[str, str], info) -> dict[str, str]:
        params = info.data.get("params", [])
        stray = [p for p in v if p not in params]
        if stray:
            raise ValueError(f"sources for fields the judge is not shown: {stray}")
        bad = [s for s in v.values() if s not in SOURCE_KINDS and not s.startswith("metadata.")]
        if bad:
            raise ValueError(f"unknown sources {bad}; use one of {list(SOURCE_KINDS)} or metadata.<key>")
        return v

    def source(self, param: str) -> str:
        """Where the judge-visible field ``param`` comes from."""
        return self.sources.get(param, DEFAULT_SOURCES[param])

    @property
    def lo(self) -> int:
        return min(b.range[0] for b in self.rubric)

    @property
    def hi(self) -> int:
        return max(b.range[1] for b in self.rubric)


class CompositeRubric(BaseModel):
    """Judge rubrics applied to the same answer and folded into one score.

    ``combine: mean`` - the score is the mean of the component scores;
    ``pass: all``    - the answer passes only if every component passes.
    """
    model_config = ConfigDict(populate_by_name=True)

    name: str
    compose: list[str] = Field(min_length=2)
    combine: Literal["mean"] = "mean"
    pass_: Literal["all"] = Field("all", alias="pass")

    def components(self) -> list[Rubric]:
        parts = [load_rubric(c) for c in self.compose]
        nested = [c for c, r in zip(self.compose, parts) if not isinstance(r, Rubric)]
        if nested:
            raise ValueError(f"{self.name}: components must be judge rubrics, got composites {nested}")
        return parts


def load_rubric(name_or_path: str) -> Rubric | CompositeRubric:
    """Load a rubric by suite name (``performance``) or by path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        text = p.read_text(encoding="utf-8")
    else:
        text = resources.files(__name__).joinpath(f"{name_or_path}.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return (CompositeRubric if "compose" in data else Rubric).model_validate(data)


def rubric_names() -> list[str]:
    """Names of the rubrics shipped in this package (``rubrics/<name>.yaml``)."""
    return sorted(p.name.removesuffix(".yaml") for p in resources.files(__name__).iterdir()
                  if p.name.endswith(".yaml"))
