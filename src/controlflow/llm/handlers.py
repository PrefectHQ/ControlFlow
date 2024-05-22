from controlflow.llm.tools import get_tool_calls
from controlflow.utilities.context import ctx
from controlflow.utilities.types import AssistantMessage, ToolMessage


class StreamHandler:
    def on_message_created(self, delta: AssistantMessage):
        pass

    def on_message_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        pass

    def on_message_done(self, message: AssistantMessage):
        pass

    def on_tool_call_created(self, delta: AssistantMessage):
        pass

    def on_tool_call_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        pass

    def on_tool_call_done(self, message: AssistantMessage):
        pass

    def on_tool_result(self, message: ToolMessage):
        pass


class AsyncStreamHandler(StreamHandler):
    async def on_message_created(self, delta: AssistantMessage):
        pass

    async def on_message_delta(
        self, delta: AssistantMessage, snapshot: AssistantMessage
    ):
        pass

    async def on_message_done(self, message: AssistantMessage):
        pass

    async def on_tool_call_created(self, delta: AssistantMessage):
        pass

    async def on_tool_call_delta(
        self, delta: AssistantMessage, snapshot: AssistantMessage
    ):
        pass

    async def on_tool_call_done(self, message: AssistantMessage):
        pass

    async def on_tool_result(self, message: ToolMessage):
        pass


class TUIHandler(AsyncStreamHandler):
    async def on_message_delta(
        self, delta: AssistantMessage, snapshot: AssistantMessage
    ) -> None:
        if tui := ctx.get("tui"):
            tui.update_message(message=snapshot)

    async def on_tool_call_delta(
        self, delta: AssistantMessage, snapshot: AssistantMessage
    ):
        if tui := ctx.get("tui"):
            for tool_call in get_tool_calls(snapshot):
                tui.update_message(message=snapshot)

    async def on_tool_result(self, message: ToolMessage):
        if tui := ctx.get("tui"):
            tui.update_tool_result(message=message)


class PrintHandler(AsyncStreamHandler):
    def on_message_created(self, delta: AssistantMessage):
        print(f"Created: {delta}\n")

    def on_message_done(self, message: AssistantMessage):
        print(f"Done: {message}\n")

    def on_tool_call_created(self, delta: AssistantMessage):
        print(f"Tool call created: {delta}\n")

    def on_tool_call_done(self, message: AssistantMessage):
        print(f"Tool call: {message}\n")

    def on_tool_result(self, message: ToolMessage):
        print(f"Tool result: {message}\n")