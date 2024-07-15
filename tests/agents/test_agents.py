import controlflow
import pytest
from controlflow.agents import Agent
from controlflow.agents.names import AGENTS
from controlflow.flows import Flow
from controlflow.instructions import instructions
from controlflow.orchestration.agent_context import AgentContext
from controlflow.tasks.task import Task
from langchain_openai import ChatOpenAI


class TestAgentInitialization:
    def test_agent_gets_random_name(self):
        agent = Agent()

        assert agent.name in AGENTS

    def test_agent_default_model(self):
        agent = Agent()

        # None indicates it will be loaded from the default model
        assert agent.model is None
        assert agent.get_model() is controlflow.defaults.model

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

    def test_stable_id(self):
        agent = Agent(name="Test Agent")
        assert agent.id == "69dd1abd"

    def test_id_includes_instructions(self):
        a1 = Agent(name="Test Agent")
        a2 = Agent(name="Test Agent", instructions="abc")
        a3 = Agent(name="Test Agent", instructions="def")
        a4 = Agent(name="Test Agent", instructions="abc", description="xyz")

        assert a1.id != a2.id != a3.id != a4.id


class TestDefaultAgent:
    def test_default_agent(self):
        assert controlflow.defaults.agent.name == "Marvin"
        assert Task("task").get_agent() is controlflow.defaults.agent

    def test_default_agent_has_no_tools(self):
        assert controlflow.defaults.agent.tools == []

    def test_default_agent_has_no_model(self):
        assert controlflow.defaults.agent.model is None

    def test_default_agent_can_be_assigned(self):
        # baseline
        assert controlflow.defaults.agent.name == "Marvin"

        new_default_agent = Agent(name="New Agent")
        controlflow.defaults.agent = new_default_agent

        assert controlflow.defaults.agent.name == "New Agent"
        assert Task("task").get_agent() is new_default_agent
        assert Task("task").get_agent().name == "New Agent"

    def test_updating_the_default_model_updates_the_default_agent_model(self):
        new_model = ChatOpenAI(model="gpt-3.5-turbo")
        controlflow.defaults.model = new_model

        new_agent = controlflow.defaults.agent
        assert new_agent.model is None
        assert new_agent.get_model() is new_model

        task = Task("task")
        assert task.get_agent().model is None
        assert task.get_agent().get_model() is new_model


class TestAgentPrompt:
    @pytest.fixture
    def agent_context(self) -> AgentContext:
        return AgentContext(agents=[Agent(name="Test Agent")], flow=Flow(), tasks=[])

    def test_default_prompt(self):
        agent = Agent()
        assert agent.prompt is None

    def test_default_template(self, agent_context):
        agent = Agent()
        prompt = agent.get_prompt(context=agent_context)
        assert prompt.startswith("# Agent")

    def test_custom_prompt(self, agent_context):
        agent = Agent(prompt="Custom Prompt")
        prompt = agent.get_prompt(context=agent_context)
        assert prompt == "Custom Prompt"

    def test_custom_templated_prompt(self, agent_context):
        agent = Agent(prompt="{{ agent.name }}", name="abc")
        prompt = agent.get_prompt(context=agent_context)
        assert prompt == "abc"
