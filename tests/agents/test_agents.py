import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

import controlflow
from controlflow.agents import Agent
from controlflow.events.base import Event
from controlflow.events.events import AgentMessage
from controlflow.instructions import instructions
from controlflow.llm.rules import AnthropicRules, LLMRules, OpenAIRules
from controlflow.orchestration.handler import Handler
from controlflow.tasks.task import Task


class TestAgentInitialization:
    def test_positional_arg(self):
        agent = Agent("talk like a pirate")
        assert agent.instructions == "talk like a pirate"

    def test_agent_default_model(self):
        agent = Agent()

        # None indicates it will be loaded from the default model
        assert agent.model is None
        assert agent.get_model() is controlflow.defaults.model

    def test_agent_model(self):
        model = ChatOpenAI(model="gpt-4o-mini")
        agent = Agent(model=model)

        # None indicates it will be loaded from the default model
        assert agent.model is model
        assert agent.get_model() is model

    def test_agent_model_from_string(self):
        agent1 = Agent(model="openai/gpt-4o-mini")
        assert isinstance(agent1.model, ChatOpenAI)
        assert agent1.model.model_name == "gpt-4o-mini"

        agent2 = Agent(model="anthropic/claude-3-haiku-20240307")
        assert isinstance(agent2.model, ChatAnthropic)
        assert agent2.model.model == "claude-3-haiku-20240307"

    def test_agent_model_with_invalid_format(self):
        with pytest.raises(ValueError, match="The model `gpt-4o` is not valid."):
            Agent(model="gpt-4o")

    def test_agent_model_from_unsupported_provider(self):
        with pytest.raises(
            ValueError, match="Could not load provider `abc` automatically"
        ):
            Agent(model="abc/def")

    def test_agent_loads_instructions_at_creation(self):
        with instructions("test instruction"):
            agent = Agent()

        assert "test instruction" in agent.instructions

    @pytest.mark.skip(reason="IDs are not stable right now")
    def test_stable_id(self):
        agent = Agent(name="Test Agent")
        assert agent.id == "69dd1abd"

    def test_id_includes_instructions(self):
        a1 = Agent()
        a2 = Agent(instructions="abc")
        a3 = Agent(instructions="def")
        a4 = Agent(instructions="abc", description="xyz")

        assert a1.id != a2.id != a3.id != a4.id


class TestDefaultAgent:
    def test_default_agent(self):
        assert controlflow.defaults.agent.name == "Marvin"
        assert Task("task").get_agents() == [controlflow.defaults.agent]

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
        assert Task("task").get_agents() == [new_default_agent]
        assert Task("task").get_agents()[0].name == "New Agent"

    def test_updating_the_default_model_updates_the_default_agent_model(self):
        new_model = ChatOpenAI(model="gpt-4o-mini")
        controlflow.defaults.model = new_model

        new_agent = controlflow.defaults.agent
        assert new_agent.model is None
        assert new_agent.get_model() is new_model

        task = Task("task")
        assert task.get_agents()[0].model is None
        assert task.get_agents()[0].get_model() is new_model


class TestAgentPrompt:
    def test_default_prompt(self):
        agent = Agent()
        assert agent.prompt is None

    def test_default_template(self):
        agent = Agent()
        prompt = agent.get_prompt()
        assert prompt.startswith("# Agent")

    def test_custom_prompt(self):
        agent = Agent(prompt="Custom Prompt")
        prompt = agent.get_prompt()
        assert prompt == "Custom Prompt"

    def test_custom_templated_prompt(self):
        agent = Agent(name="abc", prompt="{{ agent.name }}")
        prompt = agent.get_prompt()
        assert prompt == "abc"


class TestAgentSerialization:
    def test_serialize_for_prompt(self):
        agent = Agent(name="Test", description="A test agent", interactive=True)
        serialized = agent.serialize_for_prompt()
        assert serialized["name"] == "Test"
        assert serialized["description"] == "A test agent"
        assert serialized["interactive"] is True
        assert "id" in serialized
        assert "tools" in serialized

    def test_serialize_tools(self):
        def dummy_tool():
            """Dummy tool description"""
            pass

        agent = Agent(name="Test", tools=[dummy_tool])
        serialized_tools = agent._serialize_tools(agent.tools)
        assert len(serialized_tools) == 1
        assert serialized_tools[0]["name"] == "dummy_tool"
        assert serialized_tools[0]["description"] == "Dummy tool description"


class TestAgentLLMRules:
    def test_get_llm_rules(self):
        agent = Agent(name="Test")
        rules = agent.get_llm_rules()
        assert isinstance(rules, LLMRules)


class TestAgentContext:
    def test_context_manager(self):
        agent = Agent(name="Test")
        with agent:
            from controlflow.utilities.context import ctx

            assert ctx.get("agent") is agent


class TestHandlers:
    class ExampleHandler(Handler):
        def __init__(self):
            self.events = []
            self.agent_messages = []

        def on_event(self, event: Event):
            self.events.append(event)

        def on_agent_message(self, event: AgentMessage):
            self.agent_messages.append(event)

    @pytest.mark.usefixtures("default_fake_llm")
    def test_agent_run_with_handlers(self):
        handler = self.ExampleHandler()
        agent = Agent()
        agent.run(
            "Calculate 2 + 2", result_type=int, handlers=[handler], max_llm_calls=1
        )

        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("default_fake_llm")
    async def test_agent_run_async_with_handlers(self):
        handler = self.ExampleHandler()
        agent = Agent()
        await agent.run_async(
            "Calculate 2 + 2", result_type=int, handlers=[handler], max_llm_calls=1
        )

        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1


class TestLLMRules:
    def test_llm_rules_from_model_openai(self):
        agent = Agent(model=ChatOpenAI(model="gpt-4o-mini"))
        rules = agent.get_llm_rules()
        assert isinstance(rules, OpenAIRules)

    def test_llm_rules_from_model_anthropic(self):
        agent = Agent(model=ChatAnthropic(model="claude-3-haiku-20240307"))
        rules = agent.get_llm_rules()
        assert isinstance(rules, AnthropicRules)

    def test_custom_llm_rules(self):
        rules = LLMRules(model=None)
        agent = Agent(llm_rules=rules, model=ChatOpenAI(model="gpt-4o-mini"))
        assert agent.get_llm_rules() is rules
