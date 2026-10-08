import pytest
from inspect_ai.model import get_model

from tests.parity.fake_openai import FakeOpenAI


@pytest.fixture(scope="session")
def fake_openai():
    """One scripted server for the whole test session (see ``fake_openai.py``)."""
    with FakeOpenAI() as srv:
        yield srv


@pytest.fixture(scope="session")
def models(fake_openai):
    """``(model under evaluation, judge)`` bound to the scripted server.

    The base URL is passed to the model, not through ``VLLM_BASE_URL`` / ``JUDGE_BASE_URL``:
    Inspect loads ``.env`` at every eval (overriding existing variables inside VS Code), so
    environment variables would let a developer's ``.env`` redirect the tests to a real server.
    """
    url = fake_openai.base_url
    return (get_model("vllm/fake", base_url=url, api_key="x"),
            get_model("openai-api/judge/fake", base_url=url, api_key="x"))
