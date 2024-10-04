import json
import uuid
from contextlib import contextmanager
from typing import Union

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import controlflow
from controlflow.events.history import InMemoryHistory
from controlflow.llm.messages import AIMessage, BaseMessage, ToolCall
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
        messages = []

        for r in responses:
            if isinstance(r, str):
                messages.append(AIMessage(content=r))
            elif isinstance(r, dict):
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            ToolCall(name=r["name"], args=r.get("args", {}), id="")
                        ],
                    )
                )
            else:
                messages.append(r)

        if any(not isinstance(m, BaseMessage) for m in messages):
            raise ValueError(
                "Responses must be provided as either a list of strings, tool call dicts, or AIMessages. "
                "Each item in the list will be emitted in a cycle when the LLM is called."
            )

        self.responses = messages

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
