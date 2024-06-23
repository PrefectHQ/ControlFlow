from contextlib import contextmanager
from typing import Union

import langchain_core.messages
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import controlflow
from controlflow.flows.history import InMemoryHistory
from controlflow.llm.messages import BaseMessage, MessageType


class FakeLLM(FakeMessagesListChatModel):
    def set_responses(
        self, responses: list[Union[MessageType, langchain_core.messages.BaseMessage]]
    ):
        new_responses = []
        for msg in responses:
            if isinstance(msg, BaseMessage):
                msg = msg.to_langchain_message()
            new_responses.append(msg)
        self.responses = new_responses

    def bind_tools(self, *args, **kwargs):
        """When binding tools, passthrough"""
        return self

    def get_num_tokens_from_messages(self, messages: list) -> int:
        """Approximate token counter for messages"""
        return len(str(messages))


@contextmanager
def record_messages(
    remove_additional_kwargs: bool = True, remove_tool_call_chunks: bool = True
):
    """
    Context manager for recording all messages in a flow, useful for testing.


    with record_messages() as messages:
        cf.Task("say hello").run()

    assert messages[0].content == "Hello!"

    """
    history = InMemoryHistory(history={})
    old_default_history = controlflow.default_history
    controlflow.default_history = history

    messages = []

    try:
        yield messages
    finally:
        controlflow.default_history = old_default_history

        _messages_buffer = []
        for _, thread_messages in history.history.items():
            for message in thread_messages:
                message = message.copy()
                if hasattr(message, "additional_kwargs") and remove_additional_kwargs:
                    message.additional_kwargs = {}
                if hasattr(message, "tool_call_chunks") and remove_tool_call_chunks:
                    message.tool_call_chunks = []
                _messages_buffer.append(message)

        messages.extend(sorted(_messages_buffer, key=lambda m: m.timestamp))
