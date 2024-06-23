import controlflow
from controlflow.llm.messages import AgentReference, AIMessage


class TestAIMessage:
    def test_agent(self):
        agent = controlflow.Agent(name="Test Agent!")
        message = AIMessage(content="", agent=agent)
        assert isinstance(message.agent, AgentReference)
        assert message.agent.name == "Test Agent!"

    def test_name_loaded_from_agent(self):
        agent = controlflow.Agent(name="Test Agent!")
        message = AIMessage(content="", agent=agent)
        assert message.name == "Test-Agent"
