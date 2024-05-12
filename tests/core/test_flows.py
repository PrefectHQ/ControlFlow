# test_flow.py
from unittest.mock import MagicMock

from controlflow.core.agent import Agent
from controlflow.core.flow import Flow, get_flow
from controlflow.utilities.context import ctx


class TestFlow:
    def test_flow_initialization(self):
        flow = Flow()
        assert flow.thread is not None
        assert len(flow.tools) == 0
        assert len(flow.agents) == 1
        assert isinstance(flow.agents[0], Agent)
        assert len(flow.context) == 0

    def test_flow_with_custom_agents(self):
        agent1 = Agent(name="Agent 1")
        agent2 = Agent(name="Agent 2")
        flow = Flow(agents=[agent1, agent2])
        assert len(flow.agents) == 2
        assert agent1 in flow.agents
        assert agent2 in flow.agents

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

    def test_add_message(self, monkeypatch):
        flow = Flow()
        mocked_add = MagicMock()
        monkeypatch.setattr(flow.thread, "add", mocked_add)
        flow.add_message("Test message", role="user")
        mocked_add.assert_called_once_with("Test message", role="user")

    def test_flow_context_manager(self):
        with Flow() as flow:
            assert ctx.get("flow") == flow
            assert ctx.get("tasks") == []
        assert ctx.get("flow") is None
        assert ctx.get("tasks") is None

    def test_get_flow_within_context(self):
        with Flow() as flow:
            assert get_flow() == flow

    def test_get_flow_without_context(self):
        flow1 = get_flow()
        with Flow() as flow2:
            pass
        flow3 = get_flow()
        assert flow1 == flow3
        assert flow1 != flow2

    def test_get_flow_nested_contexts(self):
        with Flow() as flow1:
            assert get_flow() == flow1
            with Flow() as flow2:
                assert get_flow() == flow2
            assert get_flow() == flow1
