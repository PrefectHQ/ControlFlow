from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import litellm
import pytest
from controlflow.core.agent import Agent
from controlflow.core.task import Task, TaskStatus
from controlflow.llm.completions import Response
from controlflow.settings import temporary_settings


def new_chunk():
    chunk = litellm.ModelResponse()
    chunk.choices = [litellm.utils.StreamingChoices()]
    return chunk


@pytest.fixture
def prevent_openai_calls():
    """Prevent any calls to the OpenAI API from being made."""
    with temporary_settings(llm_api_key="unset"):
        yield


@pytest.fixture
def mock_controller_run_agent(monkeypatch, prevent_openai_calls):
    MockRunAgent = AsyncMock()
    MockThreadGetMessages = Mock()

    async def _run_agent(agent: Agent, tasks: list[Task] = None, thread=None):
        for task in tasks:
            if agent in task.get_agents():
                # we can't call mark_successful because we don't know the result
                task.status = TaskStatus.SUCCESSFUL

    MockRunAgent.side_effect = _run_agent

    def get_messages(*args, **kwargs):
        return []

    MockThreadGetMessages.side_effect = get_messages

    monkeypatch.setattr(
        "controlflow.core.controller.controller.Controller._run_agent", MockRunAgent
    )
    yield MockRunAgent


@pytest.fixture
def mock_controller_choose_agent(monkeypatch):
    MockChooseAgent = Mock()

    def choose_agent(agents, **kwargs):
        return agents[0]

    MockChooseAgent.side_effect = choose_agent

    monkeypatch.setattr(
        "controlflow.core.controller.controller.Controller.choose_agent",
        MockChooseAgent,
    )
    yield MockChooseAgent


@pytest.fixture
def mock_controller(mock_controller_choose_agent, mock_controller_run_agent):
    pass


@pytest.fixture
def mock_completion(monkeypatch):
    """
    Mock the completion function from the LLM module. Use this fixture to set
    the response value ahead of calling the completion.

    Example:

    def test_completion(mock_completion):
        mock_completion.set_response("Hello, world!")
        response = litellm.completion(...)
        assert response == "Hello, world!"
    """
    response = litellm.ModelResponse()

    def set_response(message: str):
        response.choices[0].message.content = message

    def mock_func(*args, **kwargs):
        return Response(responses=[response], messages=[])

    monkeypatch.setattr("controlflow.llm.completions.completion", mock_func)
    mock_func.set_response = set_response

    return mock_func


@pytest.fixture
def mock_completion_stream(monkeypatch):
    """
    Mock the completion function from the LLM module. Use this fixture to set
    the response value ahead of calling the completion.

    Example:

    def test_completion(mock_completion):
        mock_completion.set_response("Hello, world!")
        response = litellm.completion(...)
        assert response == "Hello, world!"
    """
    response = litellm.ModelResponse()
    chunk = litellm.ModelResponse()
    chunk.choices = [litellm.utils.StreamingChoices()]

    def set_response(message: str):
        response.choices[0].message.content = message

    def mock_func_deltas(*args, **kwargs):
        for c in response.choices[0].message.content:
            chunk = new_chunk()
            chunk.choices[0].delta.content = c
            yield chunk, response

    monkeypatch.setattr(
        "controlflow.llm.completions.completion_stream", mock_func_deltas
    )
    mock_func_deltas.set_response = set_response

    return mock_func_deltas


@pytest.fixture
def mock_completion_async(monkeypatch):
    """
    Mock the completion function from the LLM module. Use this fixture to set
    the response value ahead of calling the completion.

    Example:

    def test_completion(mock_completion):
        mock_completion.set_response("Hello, world!")
        response = litellm.completion(...)
        assert response == "Hello, world!"
    """
    response = litellm.ModelResponse()

    def set_response(message: str):
        response.choices[0].message.content = message

    async def mock_func(*args, **kwargs):
        return Response(responses=[response], messages=[])

    monkeypatch.setattr("controlflow.llm.completions.completion_async", mock_func)
    mock_func.set_response = set_response

    return mock_func


@pytest.fixture
def mock_completion_stream_async(monkeypatch):
    """
    Mock the completion function from the LLM module. Use this fixture to set
    the response value ahead of calling the completion.

    Example:

    def test_completion(mock_completion):
        mock_completion.set_response("Hello, world!")
        response = litellm.completion(...)
        assert response == "Hello, world!"
    """
    response = litellm.ModelResponse()

    def set_response(message: str):
        response.choices[0].message.content = message

    async def mock_func_deltas(*args, **kwargs):
        for c in response.choices[0].message.content:
            chunk = new_chunk()
            chunk.choices[0].delta.content = c
            yield chunk, response

    monkeypatch.setattr(
        "controlflow.llm.completions.completion_stream_async", mock_func_deltas
    )
    mock_func_deltas.set_response = set_response

    return mock_func_deltas
