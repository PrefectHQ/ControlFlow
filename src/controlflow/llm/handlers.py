import asyncio

import litellm

from controlflow.llm.tools import ToolResult
from controlflow.utilities.types import Message


class StreamHandler:
    def on_message_created(self, delta: litellm.utils.Delta):
        return asyncio.sleep(0)

    def on_message_delta(self, delta: litellm.utils.Delta, snapshot: litellm.Message):
        return asyncio.sleep(0)

    def on_message_done(self, message: Message):
        return asyncio.sleep(0)

    def on_tool_call(self, tool_call: Message):
        return asyncio.sleep(0)

    def on_tool_result(self, tool_result: ToolResult):
        return asyncio.sleep(0)


class AsyncStreamHandler(StreamHandler):
    async def on_message_created(self, delta: litellm.utils.Delta):
        pass

    async def on_message_delta(
        self, delta: litellm.utils.Delta, snapshot: litellm.Message
    ):
        pass

    async def on_message_done(self, message: Message):
        pass

    async def on_tool_call(self, tool_call: Message):
        pass

    async def on_tool_result(self, tool_result: ToolResult):
        pass


class PrintHandler(AsyncStreamHandler):
    def on_message_created(self, delta: litellm.utils.Delta):
        print(f"Created: {delta}\n")

    def on_message_delta(self, delta: litellm.utils.Delta, snapshot: litellm.Message):
        print(f"Updated: {delta}\n")

    def on_message_done(self, message: Message):
        print(f"Done: {message}\n")

    def on_tool_call(self, tool_call: Message):
        print(f"Tool call: {tool_call}\n")

    def on_tool_result(self, tool_result: ToolResult):
        print(f"Tool result: {tool_result}\n")
