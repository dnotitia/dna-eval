"""IFEval and IFBench: task settings, sample mapping, and (network) the pinned datasets."""
import importlib
import itertools

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model

from dna_eval.scorers import instruction_following as scorer_module
from dna_eval.scorers.instruction_following import check_instruction
from dna_eval.vendor.ifbench import instructions_registry

TASKS = {"ifeval": ("dna_eval.tasks.benchmarks.ifeval", "google/IFEval", 541, 1, "prompt_level_strict"),
         "ifbench": ("dna_eval.tasks.benchmarks.ifbench", "allenai/IFBench_test", 300, 8, "prompt_level_loose")}

RECORD = {"key": 7, "prompt": "Write something.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{"N": None}]}


@pytest.fixture(params=sorted(TASKS))
def module(request, monkeypatch):
    """The task module, with ``hf_dataset`` replaced by a recorder so no download happens."""
    mod = importlib.import_module(TASKS[request.param][0])
    calls = []

    def fake_hf_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return [kwargs["sample_fields"](RECORD)]

    monkeypatch.setattr(mod, "hf_dataset", fake_hf_dataset)
    mod.calls = calls
    return request.param, mod


def test_task_settings(module):
    name, mod = module
    _, repo, _, epochs, headline = TASKS[name]
    task = getattr(mod, name)()
    path, kwargs = mod.calls[0]
    assert path == repo and kwargs["revision"] == mod.DATASET_REVISION and kwargs["split"] == "train"
    assert task.epochs == epochs
    assert task.headline_metric.metric == headline
    assert task.config.max_tokens is None
    assert task.config.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}
    assert getattr(mod, name)(thinking=False, epochs=3, revision="main").epochs == 3
    assert mod.calls[-1][1]["revision"] == "main"
    assert getattr(mod, name)(thinking=False).config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_record_becomes_a_sample_with_the_instructions_in_metadata(module):
    name, mod = module
    sample = getattr(mod, name)().dataset[0]
    assert isinstance(sample, Sample)
    assert sample.id == 7 and sample.input == "Write something."
    assert sample.metadata == {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{"N": None}]}


def test_task_runs_end_to_end_on_mockllm(module, monkeypatch, tmp_path):
    from inspect_ai import eval as inspect_eval

    name, mod = module
    monkeypatch.setattr(scorer_module, "_prepare", lambda: None)
    model = get_model("mockllm/model", custom_outputs=itertools.cycle([ModelOutput.from_content("mockllm/model", "ok")]))
    log = inspect_eval(getattr(mod, name)(epochs=2), model=model, display="none", log_dir=str(tmp_path))[0]
    metrics = {k: m.value for k, m in log.results.scores[0].metrics.items()}
    assert metrics["prompt_level_strict"] == 1.0 and metrics["instruction_level_loose"] == 1.0
    assert log.eval.task_args["revision"] == mod.DATASET_REVISION


# --- the pinned datasets --------------------------------------------------------------------


@pytest.fixture(scope="module")
def nltk_ready():
    scorer_module._prepare()


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(TASKS))
def test_pinned_dataset_loads_and_every_instruction_runs(name, nltk_ready):
    """Every instruction in the data resolves to a verifier that runs on a dummy response."""
    mod = importlib.import_module(TASKS[name][0])
    _, _, size, _, _ = TASKS[name]
    dataset = getattr(mod, name)().dataset
    assert len(dataset) == size
    dummy = "This is a short dummy response.\n\nIt has two paragraphs, and some words."
    seen = set()
    for sample in dataset:
        ids, kwargs = sample.metadata["instruction_id_list"], sample.metadata["kwargs"]
        assert len(ids) == len(kwargs) and ids
        for instruction_id, kw in zip(ids, kwargs):
            assert instruction_id in instructions_registry.INSTRUCTION_DICT, instruction_id
            strict, loose = check_instruction(instruction_id, kw, sample.input, dummy)
            assert isinstance(strict, bool) and isinstance(loose, bool) and (loose or not strict)
            seen.add(instruction_id)
    assert seen


@pytest.mark.network
def test_every_registry_instruction_runs_on_a_dummy_response(nltk_ready):
    """The datasets use 25 (IFEval) and 58 (IFBench) of the 83 registered instructions; those
    in the data have just run, so here run each registered one with the arguments of a sample
    that uses it, where one exists."""
    used = {}
    for name in TASKS:
        mod = importlib.import_module(TASKS[name][0])
        for sample in getattr(mod, name)().dataset:
            for i, kw in zip(sample.metadata["instruction_id_list"], sample.metadata["kwargs"]):
                used.setdefault(i, (kw, sample.input))
    assert len(used) >= 58
    for instruction_id, (kw, prompt) in used.items():
        check_instruction(instruction_id, kw, prompt, "dummy response\nsecond line")
