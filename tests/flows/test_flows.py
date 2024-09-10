from controlflow.agents import Agent
from controlflow.events.events import UserMessage
from controlflow.flows import Flow, get_flow
from controlflow.tasks.task import Task
from controlflow.utilities.context import ctx


class TestFlowInitialization:
    def test_flow_initialization(self):
        flow = Flow()
        assert flow.thread_id is not None
        assert len(flow.tools) == 0
        assert flow.default_agent is None
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
            assert ctx.get("tasks") is None
        assert ctx.get("flow") is None
        assert ctx.get("tasks") is None

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

    def test_flow_context_resets_task_tracking(self):
        parent_task = Task("Parent task")
        with parent_task:
            assert ctx.get("tasks") == [parent_task]
            with Flow():
                assert ctx.get("tasks") is None
                nested_task = Task("Nested task")
                assert nested_task.parent is None
            assert ctx.get("tasks") == [parent_task]
        assert ctx.get("tasks") is None


class TestFlowHistory:
    def test_get_events_empty(self):
        flow = Flow()
        messages = flow.get_events()
        assert messages == []

    def test_load_parent_history(self):
        flow1 = Flow()
        flow1.add_events(
            [
                UserMessage(content="hello"),
                UserMessage(content="world"),
            ]
        )

        with flow1:
            flow2 = Flow()

        messages1 = flow1.get_events()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "world"]

        messages2 = flow2.get_events()
        assert messages1 == messages2

    def test_load_parent_history_sorts_messages(self):
        flow1 = Flow()
        flow1.add_events(
            [
                UserMessage(content="hello"),
            ]
        )

        with flow1:
            flow2 = Flow()
            flow2.add_events([UserMessage(content="world")])

        flow1.add_events([UserMessage(content="goodbye")])

        messages1 = flow1.get_events()
        assert len(messages1) == 2
        assert [m.content for m in messages1] == ["hello", "goodbye"]

        messages2 = flow2.get_events()
        assert len(messages2) == 3
        assert [m.content for m in messages2] == ["hello", "world", "goodbye"]

    def test_disable_load_parent_history(self):
        flow1 = Flow()
        flow1.add_events(
            [
                UserMessage(content="hello"),
                UserMessage(content="world"),
            ]
        )

        with flow1:
            flow2 = Flow(load_parent_events=False)

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

    def test_flow_sets_thread_id_for_history(self, tmpdir):
        f1 = Flow(thread_id="abc")
        f2 = Flow(thread_id="xyz")
        f3 = Flow(thread_id="abc")

        f1.add_events([UserMessage(content="test")])
        assert len(f1.get_events()) == 1
        assert len(f2.get_events()) == 0
        assert len(f3.get_events()) == 1


class TestFlowCreatesDefaults:
    def test_flow_with_custom_agents(self):
        agent1 = Agent()
        flow = Flow(default_agent=agent1)  # Changed from 'agent'
        assert flow.default_agent == agent1  # Changed from 'agent'

    def test_flow_agent_becomes_task_default(self):
        agent = Agent()
        t1 = Task("t1")
        assert agent not in t1.get_agents()
        assert len(t1.get_agents()) == 1

        with Flow(default_agent=agent):  # Changed from 'agent'
            t2 = Task("t2")
            assert agent in t2.get_agents()
            assert len(t2.get_agents()) == 1


class TestFlowPrompt:
    def test_default_prompt(self):
        flow = Flow()
        assert flow.prompt is None

    def test_default_template(self):
        flow = Flow()
        prompt = flow.get_prompt()
        assert prompt.startswith("# Flow")

    def test_custom_prompt(self):
        flow = Flow(prompt="Custom Prompt")
        prompt = flow.get_prompt()
        assert prompt == "Custom Prompt"

    def test_custom_templated_prompt(self):
        flow = Flow(prompt="{{ flow.name }}", name="abc")
        prompt = flow.get_prompt()
        assert prompt == "abc"
