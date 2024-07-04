import controlflow
import pytest
from controlflow.llm.messages import BaseMessage
from controlflow.utilities.testing import FakeLLM


@pytest.fixture(autouse=True)
def restore_defaults(monkeypatch):
    """
    Monkeypatch defaults to themselves, which will automatically reset them after every test
    """
    monkeypatch.setattr(controlflow.defaults, "agent", controlflow.defaults.agent)
    monkeypatch.setattr(controlflow.defaults, "model", controlflow.defaults.model)
    monkeypatch.setattr(controlflow.defaults, "history", controlflow.defaults.history)
    yield


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM(responses=[])


@pytest.fixture()
def default_fake_llm(fake_llm) -> FakeLLM:
    controlflow.defaults.model = fake_llm
    return fake_llm
