import controlflow
from controlflow.llm.messages import AIMessage


class TestAIMessage:
    def test_agent_stored_as_dict(self):
        agent = controlflow.Agent(name="Test Agent!")
        message = AIMessage(content="", agent=agent)
        assert isinstance(message.agent, dict)
        assert message.agent["name"] == "Test Agent!"

    def test_name_loaded_from_agent(self):
        agent = controlflow.Agent(name="Test Agent!")
        message = AIMessage(content="", agent=agent)
        assert message.name == "Test-Agent"
