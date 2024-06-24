from controlflow.agents import Agent
from controlflow.flows import Flow, get_flow
from controlflow.llm.messages import UserMessage
from controlflow.tasks.task import Task
from controlflow.utilities.context import ctx


class TestFlowInitialization:
    def test_flow_initialization(self):
        flow = Flow()
        assert flow.thread_id is not None
        assert len(flow.tools) == 0
        assert len(flow.agents) == 0
        assert len(flow.context) == 0

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

        assert flow.tasks == {t1.id: t1, t2.id: t2}

    def test_tasks_created_in_nested_flows_only_in_inner_flow(self):
        with Flow() as flow1:
            t1 = Task("test 1")
            with Flow() as flow2:
                t2 = Task("test 2")

        assert flow1.tasks == {t1.id: t1}
        assert flow2.tasks == {t2.id: t2}


class TestFlowHistory:
    def test_get_messages_empty(self):
        flow = Flow()
        messages = flow.get_messages()
        assert messages == []

    def test_add_messages_with_history(self):
        flow = Flow()
        flow.add_messages(
            messages=[UserMessage(content="hello"), UserMessage(content="world")]
        )
        messages = flow.get_messages()
        assert len(messages) == 2
        assert [m.content for m in messages] == ["hello", "world"]

    def test_copy_parent_history(self):
        flow1 = Flow()
        flow1.add_messages(
            messages=[UserMessage(content="hello"), UserMessage(content="world")]
        )

        with flow1:
            flow2 = Flow()

        messages1 = flow1.get_messages()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "world"]

        messages2 = flow2.get_messages()
        assert len(messages2) == 2
        assert [m.content for m in messages2] == ["hello", "world"]

    def test_disable_copying_parent_history(self):
        flow1 = Flow()
        flow1.add_messages(
            messages=[UserMessage(content="hello"), UserMessage(content="world")]
        )

        with flow1:
            flow2 = Flow(copy_parent=False)

        messages1 = flow1.get_messages()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "world"]

        messages2 = flow2.get_messages()
        assert len(messages2) == 0

    def test_child_flow_messages_dont_go_to_parent(self):
        flow1 = Flow()
        flow1.add_messages(
            messages=[UserMessage(content="hello"), UserMessage(content="world")]
        )

        with flow1:
            flow2 = Flow()
            flow2.add_messages(messages=[UserMessage(content="goodbye")])

        messages1 = flow1.get_messages()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "world"]

        messages2 = flow2.get_messages()
        assert len(messages2) == 3
        assert [m.content for m in messages2] == ["hello", "world", "goodbye"]


class TestFlowCreatesDefaults:
    def test_flow_with_custom_agents(self):
        agent1 = Agent(name="Agent 1")
        agent2 = Agent(name="Agent 2")
        flow = Flow(agents=[agent1, agent2])
        assert len(flow.agents) == 2
        assert agent1 in flow.agents
        assert agent2 in flow.agents

    def test_flow_agent_becomes_task_default(self):
        agent = Agent()
        t1 = Task("t1")
        assert t1.agents != [agent]

        with Flow(agents=[agent]):
            t2 = Task("t2")
            assert t2.get_agents() == [agent]
