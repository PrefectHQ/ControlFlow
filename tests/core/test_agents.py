import controlflow
from controlflow.core.agent import Agent, get_default_agent
from controlflow.core.agent.names import NAMES
from controlflow.core.task import Task


class TestAgentInitialization:
    def test_agent_gets_random_name(self):
        agent = Agent()

        assert agent.name in NAMES

    def test_agent_default_model(self):
        agent = Agent()

        assert agent.model is controlflow.get_default_model()


class TestDefaultAgent:
    def test_default_agent_is_marvin(self):
        agent = get_default_agent()
        assert agent.name == "Marvin"

    def test_default_agent_has_no_tools(self):
        assert get_default_agent().tools == []

    def test_default_agent_can_be_assigned(self):
        # baseline
        assert get_default_agent().name == "Marvin"

        new_default_agent = Agent(name="New Agent")
        controlflow.default_agent = new_default_agent

        assert get_default_agent().name == "New Agent"
        assert Task("task").get_agents()[0] is new_default_agent
        assert [a.name for a in Task("task").get_agents()] == ["New Agent"]

    def test_default_agent(self):
        assert get_default_agent().name == "Marvin"
        assert Task("task").get_agents()[0] is get_default_agent()
