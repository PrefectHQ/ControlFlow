import pytest
from pydantic import ValidationError

from controlflow.agents.agent import Agent
from controlflow.events.events import UserMessage
from controlflow.flows import Flow
from controlflow.orchestration.agent_context import AgentContext
from controlflow.orchestration.print_handler import PrintHandler
from controlflow.tools import tool
from controlflow.utilities.testing import SimpleTask


@pytest.fixture
def agent_context() -> AgentContext:
    return AgentContext(flow=Flow(), tasks=[SimpleTask()])


class TestAgentContextPersistEvents:
    def test_persist_event(self, agent_context: AgentContext):
        event = UserMessage(content="test")
        assert not agent_context.get_events()
        agent_context.handle_event(event=event)
        assert event in agent_context.get_events()

    def test_persist_event_false(self, agent_context: AgentContext):
        event = UserMessage(content="test", persist=False)
        assert not agent_context.get_events()
        agent_context.handle_event(event=event)
        assert event not in agent_context.get_events()

    def test_persist_event_false_kwarg(self, agent_context: AgentContext):
        event = UserMessage(content="test")
        assert not agent_context.get_events()
        agent_context.handle_event(event=event, persist=False)
        assert event not in agent_context.get_events()

    def test_persist_event_false_but_kwarg_true(self, agent_context: AgentContext):
        event = UserMessage(content="test", persist=False)
        assert not agent_context.get_events()
        agent_context.handle_event(event=event, persist=True)
        assert event in agent_context.get_events()


class TestAgentContextHandler:
    def test_add_handlers(self, agent_context: AgentContext):
        handler = PrintHandler()
        assert not agent_context.handlers
        agent_context.add_handlers([handler])
        assert handler in agent_context.handlers

    def test_add_handlers_validate(self, agent_context: AgentContext):
        with pytest.raises(ValidationError):
            agent_context.add_handlers([1, 2, 3])


class TestAgentContextCompilePrompt:
    def test_instructions_in_prompt(self, agent_context):
        agent_context.add_instructions(["custom instructions!"])
        prompt = agent_context.compile_prompt(agent=Agent("test-agent"))
        assert "custom instructions!" in prompt

    def test_tool_instructions_in_prompt(self, agent_context):
        @tool(instructions="custom tool instructions!")
        def f():
            pass

        agent_context.add_tools([f])
        prompt = agent_context.compile_prompt(agent=Agent("test-agent"))
        assert "custom tool instructions!" in prompt
