import pydantic
import pytest
from langchain_openai import ChatOpenAI

import controlflow
import controlflow.events.history
import controlflow.llm


def test_default_model_failed_validation():
    with pytest.raises(
        pydantic.ValidationError,
        match="Input must be an instance of dict or BaseChatModel",
    ):
        controlflow.defaults.model = 5


def test_set_default_model():
    model = ChatOpenAI(temperature=0.1)
    controlflow.defaults.model = model
    assert controlflow.Agent().get_model() is model


def test_default_agent_failed_validation():
    with pytest.raises(
        pydantic.ValidationError,
        match="Input should be a valid dictionary or instance of Agent",
    ):
        controlflow.defaults.agent = 5


def test_set_default_agent():
    agent = controlflow.Agent()
    controlflow.defaults.agent = agent
    assert controlflow.Task("").get_agents() == [agent]


def test_default_history_failed_validation():
    with pytest.raises(
        pydantic.ValidationError,
        match="Input should be a valid dictionary or instance of History",
    ):
        controlflow.defaults.history = 5


def test_set_default_history():
    history = controlflow.events.history.InMemoryHistory()
    controlflow.defaults.history = history
    assert controlflow.Flow().history is history
