from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from controlflow.core.agent import Agent
from controlflow.core.task import Task, TaskStatus
from marvin.settings import temporary_settings as temporary_marvin_settings


@pytest.fixture
def prevent_openai_calls():
    """Prevent any calls to the OpenAI API from being made."""
    with temporary_marvin_settings(openai__api_key="unset"):
        yield


@pytest.fixture
def mock_run(monkeypatch, prevent_openai_calls):
    """
    This fixture mocks the calls to the OpenAI Assistants API. Use it in a test
    and assign any desired side effects (like completing a task) to the mock
    object's `.side_effect` attribute.

    For example:

    def test_example(mock_run):
        task = Task(objective="Say hello")

        def side_effect():
            task.mark_complete()

        mock_run.side_effect = side_effect

        task.run()

    """
    MockRun = AsyncMock()
    monkeypatch.setattr("controlflow.core.controller.controller.Run.run_async", MockRun)
    yield MockRun


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
    monkeypatch.setattr(
        "marvin.beta.assistants.Thread.get_messages", MockThreadGetMessages
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
