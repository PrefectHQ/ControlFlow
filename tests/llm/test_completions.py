from unittest.mock import Mock

from controlflow.llm.completions import completion
from controlflow.llm.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel


def test_completions_replace_invalid_agent_names(fake_llm, monkeypatch):
    mock_invoke = Mock(return_value=AIMessage(content=""))
    monkeypatch.setattr(BaseChatModel, "invoke", mock_invoke)
    messages = [AIMessage(content="", name="An invalid.name!")]
    completion(messages=messages, model=fake_llm)
    # the name was modified before being sent to the model
    mock_invoke.call_args[1]["input"][0].name == "An-invalid-name-"
    # the name was not modified on the original message
    assert messages[0].name == "An invalid.name!"
