from contextlib import contextmanager

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import controlflow
from controlflow.events.history import InMemoryHistory
from controlflow.llm.messages import BaseMessage


class FakeLLM(FakeMessagesListChatModel):
    def set_responses(self, responses: list[BaseMessage]):
        self.responses = responses

    def bind_tools(self, *args, **kwargs):
        """When binding tools, passthrough"""
        return self

    def get_num_tokens_from_messages(self, messages: list) -> int:
        """Approximate token counter for messages"""
        return len(str(messages))


@contextmanager
def record_events():
    """
    Context manager for recording all messages in a flow, useful for testing.


    with record_events() as events:
        cf.Task("say hello").run()

    assert events[0].content == "Hello!"

    """
    history = InMemoryHistory(history={})
    old_default_history = controlflow.default_history
    controlflow.default_history = history

    events = []

    try:
        yield events
    finally:
        controlflow.default_history = old_default_history

        _events_buffer = []
        for _, thread_events in history.history.items():
            for event in thread_events:
                event = event.copy()
                _events_buffer.append(event)

        events.extend(sorted(_events_buffer, key=lambda m: m.timestamp))
