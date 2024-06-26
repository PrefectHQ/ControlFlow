import controlflow
from controlflow.agents import Agent, get_default_agent
from controlflow.agents.names import NAMES
from controlflow.instructions import instructions
from controlflow.tasks.task import Task
from langchain_openai import ChatOpenAI


class TestAgentInitialization:
    def test_agent_gets_random_name(self):
        agent = Agent()

        assert agent.name in NAMES

    def test_agent_default_model(self):
        agent = Agent()

        # None indicates it will be loaded from the default model
        assert agent.model is None
        assert agent.get_model() is controlflow.get_default_model()

    def test_agent_model(self):
        model = ChatOpenAI(model="gpt-3.5-turbo")
        agent = Agent(model=model)

        # None indicates it will be loaded from the default model
        assert agent.model is model
        assert agent.get_model() is model

    def test_agent_loads_instructions_at_creation(self):
        with instructions("test instruction"):
            agent = Agent()

        assert "test instruction" in agent.instructions


class TestDefaultAgent:
    def test_default_agent(self):
        assert get_default_agent().name == "Marvin"
        assert Task("task").get_agents()[0] is get_default_agent()

    def test_default_agent_has_no_tools(self):
        assert get_default_agent().tools == []

    def test_default_agent_has_no_model(self):
        assert get_default_agent().model is None

    def test_default_agent_can_be_assigned(self):
        # baseline
        assert get_default_agent().name == "Marvin"

        new_default_agent = Agent(name="New Agent")
        controlflow.default_agent = new_default_agent

        assert get_default_agent().name == "New Agent"
        assert Task("task").get_agents()[0] is new_default_agent
        assert [a.name for a in Task("task").get_agents()] == ["New Agent"]

    def test_updating_the_default_model_updates_the_default_agent_model(self):
        new_model = ChatOpenAI(model="gpt-3.5-turbo")
        controlflow.default_model = new_model

        new_agent = get_default_agent()
        assert new_agent.model is None
        assert new_agent.get_model() is new_model

        task = Task("task")
        assert task.get_agents()[0].model is None
        assert task.get_agents()[0].get_model() is new_model
