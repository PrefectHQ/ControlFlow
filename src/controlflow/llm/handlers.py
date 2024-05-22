import datetime

import litellm

from controlflow.llm.tools import ToolResult, get_tool_calls
from controlflow.utilities.context import ctx
from controlflow.utilities.types import Message


class StreamHandler:
    def on_message_created(self, delta: litellm.ModelResponse):
        pass

    def on_message_delta(
        self, delta: litellm.ModelResponse, snapshot: litellm.ModelResponse
    ):
        pass

    def on_message_done(self, response: litellm.ModelResponse):
        pass

    def on_tool_call_created(self, delta: litellm.ModelResponse):
        pass

    def on_tool_call_delta(
        self, delta: litellm.ModelResponse, snapshot: litellm.ModelResponse
    ):
        pass

    def on_tool_call_done(self, tool_call: Message):
        pass

    def on_tool_result(self, tool_result: ToolResult):
        pass


class AsyncStreamHandler(StreamHandler):
    async def on_message_created(self, delta: litellm.ModelResponse):
        pass

    async def on_message_delta(
        self, delta: litellm.ModelResponse, snapshot: litellm.ModelResponse
    ):
        pass

    async def on_message_done(self, response: litellm.ModelResponse):
        pass

    async def on_tool_call_created(self, delta: litellm.ModelResponse):
        pass

    async def on_tool_call_delta(
        self, delta: litellm.ModelResponse, snapshot: litellm.ModelResponse
    ):
        pass

    async def on_tool_call_done(self, tool_call: Message):
        pass

    async def on_tool_result(self, tool_result: ToolResult):
        pass


class TUIHandler(AsyncStreamHandler):
    async def on_message_delta(
        self, delta: litellm.ModelResponse, snapshot: litellm.ModelResponse
    ) -> None:
        if tui := ctx.get("tui"):
            tui.update_message(
                m_id=snapshot.id,
                message=snapshot.choices[0].message.content,
                role=snapshot.choices[0].message.role,
                timestamp=datetime.datetime.fromtimestamp(snapshot.created),
            )

    async def on_tool_call_delta(
        self, delta: litellm.ModelResponse, snapshot: litellm.ModelResponse
    ):
        if tui := ctx.get("tui"):
            for tool_call in get_tool_calls(snapshot):
                tui.update_tool_call(
                    t_id=snapshot.id,
                    tool_name=tool_call.function.name,
                    tool_args=tool_call.function.arguments,
                    timestamp=datetime.datetime.fromtimestamp(snapshot.created),
                )

    async def on_tool_result(self, message: Message):
        if tui := ctx.get("tui"):
            tui.update_tool_result(
                t_id=message.tool_result.tool_call_id,
                tool_name=message.tool_result.tool_name,
                tool_result=message.content,
                timestamp=datetime.datetime.now(),
            )


class PrintHandler(AsyncStreamHandler):
    def on_message_created(self, delta: litellm.ModelResponse):
        print(f"Created: {delta}\n")

    def on_message_done(self, response: litellm.ModelResponse):
        print(f"Done: {response}\n")

    def on_tool_call_created(self, delta: litellm.ModelResponse):
        print(f"Tool call created: {delta}\n")

    def on_tool_call_done(self, tool_call: Message):
        print(f"Tool call: {tool_call}\n")

    def on_tool_result(self, tool_result: ToolResult):
        print(f"Tool result: {tool_result}\n")
