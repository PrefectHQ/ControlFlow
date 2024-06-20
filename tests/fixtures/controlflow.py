import controlflow
import pytest
from controlflow.llm.messages import MessageType
from controlflow.utilities.testing import FakeLLM


@pytest.fixture(autouse=True)
def restore_defaults(monkeypatch):
    """
    Monkeypatch defaults to themselves, which will automatically reset them after every test
    """
    monkeypatch.setattr(controlflow, "default_agent", controlflow.default_agent)
    monkeypatch.setattr(controlflow, "default_model", controlflow.default_model)
    monkeypatch.setattr(controlflow, "default_history", controlflow.default_history)
    yield


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM(responses=[])


@pytest.fixture()
def default_fake_llm(fake_llm) -> FakeLLM:
    controlflow.default_model = fake_llm
    return fake_llm
