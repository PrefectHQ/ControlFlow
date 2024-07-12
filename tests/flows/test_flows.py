import pytest
from controlflow.agents import Agent
from controlflow.events.base import Event
from controlflow.events.events import UserMessage
from controlflow.flows import Flow, get_flow
from controlflow.orchestration.agent_context import AgentContext
from controlflow.tasks.task import Task
from controlflow.utilities.context import ctx
from controlflow.utilities.testing import SimpleTask


class TestFlowInitialization:
    def test_flow_initialization(self):
        flow = Flow()
        assert flow.thread_id is not None
        assert len(flow.tools) == 0
        assert flow.agent is None
        assert flow.context == {}

    def test_flow_with_custom_tools(self):
        def tool1():
            pass

        def tool2():
            pass

        flow = Flow(tools=[tool1, tool2])
        assert len(flow.tools) == 2
        assert tool1 in flow.tools
        assert tool2 in flow.tools

    def test_flow_with_custom_context(self):
        flow = Flow(context={"key": "value"})
        assert len(flow.context) == 1
        assert flow.context["key"] == "value"


class TestFlowContext:
    def test_flow_context_manager(self):
        with Flow() as flow:
            assert ctx.get("flow") == flow
            assert ctx.get("tasks") == []
        assert ctx.get("flow") is None
        assert ctx.get("tasks") == []

    def test_get_flow_within_context(self):
        with Flow() as flow:
            assert get_flow() == flow

    def test_get_flow_without_context(self):
        assert get_flow() is None

    def test_reentrant_flow_context(self):
        flow = Flow()
        with flow:
            assert get_flow() is flow
            with flow:
                assert get_flow() is flow
                with flow:
                    assert get_flow() is flow
                assert get_flow() is flow
            assert get_flow() is flow
        assert get_flow() is None

    def test_get_flow_nested_contexts(self):
        with Flow() as flow1:
            assert get_flow() == flow1
            with Flow() as flow2:
                assert get_flow() == flow2
            assert get_flow() == flow1
        assert get_flow() is None

    def test_tasks_created_in_flow_context(self):
        with Flow() as flow:
            t1 = Task("test 1")
            t2 = Task("test 2")

        assert flow.tasks == [t1, t2]

    def test_tasks_created_in_nested_flows_only_in_inner_flow(self):
        with Flow() as flow1:
            t1 = Task("test 1")
            with Flow() as flow2:
                t2 = Task("test 2")

        assert flow1.tasks == [t1]
        assert flow2.tasks == [t2]

    def test_inner_flow_includes_completed_parent_tasks(self):
        with Flow() as flow1:
            t1 = Task("test 1", status="SUCCESSFUL")
            t2 = Task("test 2")
            with Flow() as flow2:
                t3 = Task("test 3")

        assert flow1.tasks == [t1, t2]
        assert flow2.tasks == [t1, t3]


class TestFlowHistory:
    def test_get_events_empty(self):
        flow = Flow()
        messages = flow.get_events()
        assert messages == []

    def test_disable_copying_parent_history(self):
        flow1 = Flow()
        flow1.add_events(
            [
                UserMessage(content="hello"),
                UserMessage(content="world"),
            ]
        )

        with flow1:
            flow2 = Flow(copy_parent=False)

        messages1 = flow1.get_events()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "world"]

        messages2 = flow2.get_events()
        assert len(messages2) == 0

    def test_child_flow_messages_dont_go_to_parent(self):
        flow1 = Flow()
        flow1.add_events(
            [
                UserMessage(content="hello"),
                UserMessage(content="world"),
            ]
        )

        with flow1:
            flow2 = Flow()
            flow2.add_events([UserMessage(content="goodbye")])

        messages1 = flow1.get_events()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "world"]

        messages2 = flow2.get_events()
        assert len(messages2) == 3
        assert [m.content for m in messages2] == ["hello", "world", "goodbye"]


class TestFlowCreatesDefaults:
    def test_flow_with_custom_agents(self):
        agent1 = Agent(name="Agent 1")
        flow = Flow(agent=agent1)
        assert flow.agent == agent1

    def test_flow_agent_becomes_task_default(self):
        agent = Agent()
        t1 = Task("t1")
        assert t1.agent is not agent

        with Flow(agent=agent):
            t2 = Task("t2")
            assert t2.get_agent() == agent


class TestFlowPrompt:
    @pytest.fixture
    def agent_context(self) -> AgentContext:
        return AgentContext(agent=Agent(name="Test Agent"), flow=Flow(), tasks=[])

    def test_default_prompt(self):
        flow = Flow()
        assert flow.prompt is None

    def test_default_template(self, agent_context):
        flow = Flow()
        prompt = flow.get_prompt(context=agent_context)
        assert prompt.startswith("# Flow")

    def test_custom_prompt(self, agent_context):
        flow = Flow(prompt="Custom Prompt")
        prompt = flow.get_prompt(context=agent_context)
        assert prompt == "Custom Prompt"

    def test_custom_templated_prompt(self, agent_context):
        flow = Flow(prompt="{{ flow.name }}", name="abc")
        prompt = flow.get_prompt(context=agent_context)
        assert prompt == "abc"


class TestFlowEvents:
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
        [t1, t2, t3, t4, t5] = tasks

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

    def test_get_events_by_task(self, agents: list[Agent], flow, tasks: list[Task]):
        [t1, t2, t3, t4, t5] = tasks

        for t in [t1, t2, t3, t4, t5]:
            events = flow.get_events(tasks=[t])
            assert len(events) == 1

            events = flow.get_events(tasks=flow.graph.upstream_tasks([t]))
            assert len(events) == len(flow.graph.upstream_tasks([t]))

    def test_get_events_by_agent(self, agents: list[Agent], flow):
        a1, a2 = agents
        events = flow.get_events(agents=[a1])
        assert len(events) == 4

        events = flow.get_events(agents=[a2])
        assert len(events) == 2

    def test_get_events_by_agent_and_task(self, agents, flow, tasks: list[Task]):
        a1, a2 = agents
        [t1, t2, t3, t4, t5] = tasks
        events = flow.get_events(agents=[a1], tasks=[t1])
        assert len(events) == 1

        events = flow.get_events(agents=[a2], tasks=[t1])
        assert len(events) == 0
