from controlflow.events.base import Event
from controlflow.events.events import AgentMessage
from controlflow.orchestration.handler import Handler
from controlflow.run import run, run_async


class TestHandlers:
    class ExampleHandler(Handler):
        def __init__(self):
            self.events = []
            self.agent_messages = []

        def on_event(self, event: Event):
            self.events.append(event)

        def on_agent_message(self, event: AgentMessage):
            self.agent_messages.append(event)

    def test_run_with_handlers(self, default_fake_llm):
        handler = self.ExampleHandler()
        run("what's 2 + 2", result_type=int, handlers=[handler], max_llm_calls=1)
        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1

    async def test_run_async_with_handlers(self, default_fake_llm):
        handler = self.ExampleHandler()
        await run_async(
            "what's 2 + 2", result_type=int, handlers=[handler], max_llm_calls=1
        )

        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1


def test_run():
    result = run("what's 2 + 2", result_type=int)
    assert result == 4


async def test_run_async():
    result = await run_async("what's 2 + 2", result_type=int)
    assert result == 4
