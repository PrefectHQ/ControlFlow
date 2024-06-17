import controlflow
import pytest
from controlflow.llm.messages import MessageType
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


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
def fake_llm() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[])


@pytest.fixture()
def default_fake_llm(fake_llm, restore_defaults) -> FakeMessagesListChatModel:
    controlflow.default_agent = fake_llm
    return fake_llm
