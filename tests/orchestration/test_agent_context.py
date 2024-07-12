import pytest
from controlflow.agents import Agent
from controlflow.events.events import UserMessage
from controlflow.flows import Flow
from controlflow.orchestration.agent_context import AgentContext
from controlflow.utilities.testing import SimpleTask


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
        assert not agent_context.handlers
        agent_context.add_handlers([1, 2, 3])
        assert agent_context.handlers == [1, 2, 3]
