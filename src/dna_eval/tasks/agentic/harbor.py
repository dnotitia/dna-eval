"""Harbor-format datasets as tasks: the builder Terminal-Bench, SWE-bench and DeepSWE share.

``inspect_harbor`` is imported inside ``harbor_task``, so this module imports without it.
"""
from __future__ import annotations

import importlib.util
from importlib import metadata as _metadata

from inspect_ai import Epochs, Task

from dna_eval.scorers.harbor_reward import harbor_reward
from dna_eval.solvers.agent_scaffold import SCAFFOLDS, agent_solver
from dna_eval.tasks.agentic import HEADLINE, agentic_generate_config
from dna_eval.utils.optional import AGENTIC_INSTALL, require


def harbor_task(*, package: str, digest: str, scaffold: str, max_steps: int, command_timeout: int | None,
                epochs: int, thinking: bool, sandbox_type: str, task_names: str | list[str] | None = None,
                n_tasks: int | None = None, metadata: dict | None = None) -> Task:
    """A task over a Harbor dataset, pinned by ``digest``, run by ``scaffold`` and scored by the
    dataset's own verifier (``harbor_reward``).

    The samples are the ones ``inspect_harbor`` builds from the dataset: the instruction as the
    input and a sandbox from the task's own environment definition (a Docker-compatible
    provider; ``sandbox_type`` names it, and ``inspect eval --sandbox`` overrides it). ``epochs``
    runs are averaged. ``task_names`` (glob patterns on ``org/name``) and ``n_tasks`` select a
    subset, for a smoke test. ``metadata`` is added to what the log records about the task
    (the dataset, scaffold, limits and versions of the packages used).
    """
    require("inspect_harbor", python=(3, 12), install=AGENTIC_INSTALL)
    if scaffold != "react":
        require("inspect_swe", install=AGENTIC_INSTALL)
    if scaffold not in SCAFFOLDS:
        raise ValueError(f"unknown scaffold {scaffold!r}; choose one of {', '.join(SCAFFOLDS)}")
    from inspect_harbor import harbor

    dataset = f"{package}@{digest}"
    loaded = harbor(package_name=package, package_ref=digest, dataset_task_names=task_names, n_tasks=n_tasks,
                    sandbox_env_name=sandbox_type)
    versions = {name: _metadata.version(name.replace("_", "-")) for name in ("inspect_harbor", "inspect_swe")
                if importlib.util.find_spec(name) is not None}
    return Task(
        dataset=loaded.dataset,
        solver=agent_solver(scaffold, max_steps, command_timeout),
        scorer=harbor_reward(dataset=dataset, scaffold=scaffold),
        epochs=Epochs(epochs, "mean"),
        config=agentic_generate_config(thinking),
        headline_metric=HEADLINE,
        metadata={"dataset": dataset, "scaffold": scaffold, "max_steps": max_steps,
                  "command_timeout": command_timeout, "packages": versions, **(metadata or {})},
    )
