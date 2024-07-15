import pytest
from controlflow.agents import Agent
from controlflow.events.base import Event
from controlflow.events.events import UserMessage
from controlflow.flows import Flow
from controlflow.orchestration.agent_context import AgentContext
from controlflow.orchestration.print_handler import PrintHandler
from controlflow.tasks.task import Task
from controlflow.utilities.testing import SimpleTask
from pydantic import ValidationError


@pytest.fixture
def agent_context() -> AgentContext:
    return AgentContext(flow=Flow(), tasks=[SimpleTask()], agents=[Agent()])


class TestAgentContextPersistEvents:
    def test_persist_event(self, agent_context: AgentContext):
        event = UserMessage(content="test")
        assert not agent_context.get_visible_events()
        agent_context.handle_event(event=event)
        assert event in agent_context.get_visible_events()

    def test_persist_event_false(self, agent_context: AgentContext):
        event = UserMessage(content="test", persist=False)
        assert not agent_context.get_visible_events()
        agent_context.handle_event(event=event)
        assert event not in agent_context.get_visible_events()

    def test_persist_event_false_kwarg(self, agent_context: AgentContext):
        event = UserMessage(content="test")
        assert not agent_context.get_visible_events()
        agent_context.handle_event(event=event, persist=False)
        assert event not in agent_context.get_visible_events()

    def test_persist_event_false_but_kwarg_true(self, agent_context: AgentContext):
        event = UserMessage(content="test", persist=False)
        assert not agent_context.get_visible_events()
        agent_context.handle_event(event=event, persist=True)
        assert event in agent_context.get_visible_events()


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
        agent = Agent(name="a new test agent")
        assert agent not in agent_context.agents
        agent_context.add_agent(agent)
        assert agent in agent_context.agents

    def test_add_agents_validate(self, agent_context: AgentContext):
        with pytest.raises(ValidationError):
            agent_context.add_agent(1)


class TestAgentContextGetVisibleEvents:
    @pytest.fixture
    def agents(self):
        return [Agent(name="a1"), Agent(name="a2")]

    @pytest.fixture
    def flow(self):
        with Flow() as flow:
            t1 = SimpleTask()

            # t1 -> t2 -> t3
            t2 = SimpleTask(depends_on=[t1])
            t3 = SimpleTask(depends_on=[t2])  # noqa

            # t1 -> t4
            t4 = SimpleTask(depends_on=[t1])  # noqa

            # t5
            t5 = SimpleTask()  # noqa

        return flow

    @pytest.fixture
    def tasks(self, flow):
        return list(sorted(flow.tasks, key=lambda t: t.created_at))

    @pytest.fixture
    def events(self, agents: list[Agent], flow, tasks: list[Task]):
        a1, a2 = agents
        t1, t2, t3, t4, t5 = tasks

        events = [
            Event(event="test", task_ids=[t1.id], agent_ids=[a1.id]),
            Event(event="test", task_ids=[t2.id], agent_ids=[a1.id, a2.id]),
            Event(event="test", task_ids=[t3.id], agent_ids=[a1.id]),
            Event(event="test", task_ids=[t4.id], agent_ids=[a1.id]),
            Event(event="test", task_ids=[t5.id], agent_ids=[a2.id]),
        ]

        return events

    @pytest.fixture(autouse=True)
    def add_events(self, flow, events):
        flow.add_events(events)

    def test_get_events_by_task_ALL(self, agents: list[Agent], flow, tasks: list[Task]):
        a1, a2 = agents
        t1, t2, t3, t4, t5 = tasks

        for t in [t1, t2, t3, t4, t5]:
            context = AgentContext(flow=flow, tasks=[t])
            events = context.get_visible_events(agent=a1)
            assert len(events) == 5

    def test_get_events_by_task_UPSTREAM(
        self, agents: list[Agent], flow, tasks: list[Task]
    ):
        a1, a2 = agents
        a1.history_visibility = "UPSTREAM"
        t1, t2, t3, t4, t5 = tasks

        for t in [t1, t2, t3, t4, t5]:
            context = AgentContext(flow=flow, tasks=[t])
            events = context.get_visible_events(agent=a1)
            assert len(events) == len(flow.graph.upstream_tasks([t]))

    def test_get_events_by_task_CURRENT_TASK(
        self, agents: list[Agent], flow, tasks: list[Task]
    ):
        a1, a2 = agents
        a1.history_visibility = "CURRENT_TASK"
        t1, t2, t3, t4, t5 = tasks

        for t in [t1, t2, t3, t4, t5]:
            context = AgentContext(flow=flow, tasks=[t])
            events = context.get_visible_events(agent=a1)
            assert len(events) == 1

    def test_get_events_by_task_CURRENT_AGENT(
        self, agents: list[Agent], flow, tasks: list[Task]
    ):
        a1, a2 = agents
        a2.history_visibility = "CURRENT_AGENT"
        t1, t2, t3, t4, t5 = tasks

        for t in [t1, t2, t3, t4, t5]:
            context = AgentContext(flow=flow, tasks=[t], agents=[a2])
            events = context.get_visible_events(agent=a2)
            assert len(events) == 2

    def test_get_events_by_agent(self, agents: list[Agent], flow):
        a1, a2 = agents
        a1.history_visibility = "UPSTREAM"
        a2.history_visibility = "UPSTREAM"

        context = AgentContext(flow=flow, tasks=[], agents=[a1])
        events = context.get_visible_events(agent=a1)
        assert len(events) == 4

        context = AgentContext(flow=flow, tasks=[], agents=[a2])
        events = context.get_visible_events(agent=a2)
        assert len(events) == 2

    def test_get_events_by_agent_and_task(self, agents, flow, tasks: list[Task]):
        a1, a2 = agents
        a1.history_visibility = "UPSTREAM"
        a2.history_visibility = "UPSTREAM"
        t1, t2, t3, t4, t5 = tasks

        context = AgentContext(flow=flow, agents=[a1], tasks=[t1])
        events = context.get_visible_events(agent=a1)
        assert len(events) == 1

        context = AgentContext(flow=flow, agents=[a2], tasks=[t1])
        events = context.get_visible_events(agent=a2)
        assert len(events) == 0

        context = AgentContext(flow=flow, agents=[a1], tasks=[t2, t4])
        assert len(context.get_visible_events(agent=a1)) == 3
