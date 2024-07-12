import pytest
from controlflow.agents import Agent
from controlflow.events.events import UserMessage
from controlflow.flows import Flow
from controlflow.orchestration.agent_context import AgentContext
from controlflow.orchestration.print_handler import PrintHandler
from controlflow.utilities.testing import SimpleTask
from pydantic import ValidationError


@pytest.fixture
def agent_context() -> AgentContext:
    return AgentContext(flow=Flow(), tasks=[SimpleTask()], agents=[Agent()])


class TestAgentContextEvents:
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


class TestAgentContextAgents:
    def test_add_agents(self, agent_context: AgentContext):
        agent = Agent()
        assert agent not in agent_context.agents
        agent_context.add_agent(agent)
        assert agent in agent_context.agents

    def test_add_agents_validate(self, agent_context: AgentContext):
        with pytest.raises(ValidationError):
            agent_context.add_agent(1)
