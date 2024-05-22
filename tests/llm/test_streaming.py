from collections import Counter

import litellm
from controlflow.llm.completions import completion_stream
from controlflow.llm.handlers import StreamHandler
from controlflow.llm.tools import ToolResult
from controlflow.utilities.types import Message
from pydantic import BaseModel


class StreamCall(BaseModel):
    method: str
    args: dict


class MockStreamHandler(StreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[StreamCall] = []

    def on_message_created(self, delta: litellm.utils.Delta):
        self.calls.append(
            StreamCall(method="on_message_created", args=dict(delta=delta))
        )

    def on_message_delta(self, delta: litellm.utils.Delta, snapshot: litellm.Message):
        self.calls.append(
            StreamCall(
                method="on_message_delta", args=dict(delta=delta, snapshot=snapshot)
            )
        )

    def on_message_done(self, message: Message):
        self.calls.append(
            StreamCall(method="on_message_done", args=dict(message=message))
        )

    def on_tool_call_done(self, tool_call: ToolResult):
        self.calls.append(
            StreamCall(method="on_tool_call", args=dict(tool_call=tool_call))
        )


class TestStreamHandler:
    def test_stream(self):
        handler = MockStreamHandler()
        gen = completion_stream(messages=[{"text": "Hello"}])
        handler.stream(gen)

        method_counts = Counter(call.method for call in handler.calls)
        assert method_counts["on_message_created"] == 1
        assert method_counts["on_message_delta"] == 4
        assert method_counts["on_message_done"] == 1
