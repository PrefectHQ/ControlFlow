from contextlib import contextmanager
from typing import Union

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import controlflow
from controlflow.events.history import InMemoryHistory
from controlflow.llm.messages import AIMessage, BaseMessage
from controlflow.tasks.task import Task

COUNTER = 0


def SimpleTask(**kwargs):
    global COUNTER
    COUNTER += 1

    kwargs.setdefault("objective", "test")
    kwargs.setdefault("result_type", None)
    kwargs.setdefault("context", {})["__counter__"] = str(COUNTER)

    return Task(**kwargs)


class FakeLLM(FakeMessagesListChatModel):
    def __init__(self, *, responses: list[Union[str, BaseMessage]] = None, **kwargs):
        super().__init__(responses=[], **kwargs)
        self.set_responses(responses or ["Hello! This is a response from the FakeLLM."])

    def set_responses(self, responses: list[Union[str, BaseMessage]]):
        if any(not isinstance(m, (str, BaseMessage)) for m in responses):
            raise ValueError(
                "Responses must be provided as either a list of strings or AIMessages. "
                "Each item in the list will be emitted in a cycle when the LLM is called."
            )

        responses = [
            AIMessage(content=m) if isinstance(m, str) else m for m in responses
        ]
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
    old_default_history = controlflow.defaults.history
    controlflow.defaults.history = history

    events = []

    try:
        yield events
    finally:
        controlflow.defaults.history = old_default_history

        _events_buffer = []
        for _, thread_events in history.history.items():
            for event in thread_events:
                event = event.copy()
                _events_buffer.append(event)

        events.extend(sorted(_events_buffer, key=lambda m: m.timestamp))
