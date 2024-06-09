from rich.console import Group
from rich.live import Live

import controlflow
from controlflow.llm.formatting import format_message
from controlflow.llm.messages import (
    AIMessage,
    AIMessageChunk,
    MessageType,
    ToolMessage,
)
from controlflow.utilities.context import ctx
from controlflow.utilities.types import ControlFlowModel


class CompletionEvent(ControlFlowModel):
    type: str
    payload: dict


class CompletionHandler:
    def on_event(self, event: CompletionEvent):
        method = getattr(self, f"on_{event.type}", None)
        if not method:
            raise ValueError(f"Unknown event type: {event.type}")
        method(**event.payload)
        if event.type in ["message_done", "tool_call_done", "tool_result_done"]:
            self.on_response_message(event.payload["message"])

    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_exception(self, exc: Exception):
        pass

    def on_message_created(self, delta: AIMessageChunk):
        pass

    def on_message_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        pass

    def on_message_done(self, message: AIMessage):
        pass

    def on_tool_call_created(self, delta: AIMessageChunk):
        pass

    def on_tool_call_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        pass

    def on_tool_call_done(self, message: AIMessage):
        pass

    def on_tool_result_done(self, message: ToolMessage):
        pass

    def on_response_message(self, message: MessageType):
        """
        This handler is called whenever a message is generated that should be
        included in the completion history (e.g. a `message`, `tool_call` or
        `tool_result`). Note that this is called *in addition* to the respective
        on_*_done handlers, and can be used to quickly collect all messages
        generated during a completion.
        """
        pass


class ResponseHandler(CompletionHandler):
    """
    A handler for collecting response messages.
    """

    def __init__(self):
        self.response_messages = []

    def on_response_message(self, message: MessageType):
        self.response_messages.append(message)


class TUIHandler(CompletionHandler):
    def on_message_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk) -> None:
        if tui := ctx.get("tui"):
            tui.update_message(message=snapshot)

    def on_tool_call_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        if tui := ctx.get("tui"):
            tui.update_message(message=snapshot)

    def on_tool_result_done(self, message: ToolMessage):
        if tui := ctx.get("tui"):
            tui.update_tool_result(message=message)


class PrintHandler(CompletionHandler):
    def __init__(self):
        self.width = controlflow.settings.print_handler_width
        self.messages: dict[str, MessageType] = {}
        self.live = Live(auto_refresh=False)

    def on_start(self):
        self.live.start()

    def on_end(self):
        self.live.stop()

    def on_exception(self, exc: Exception):
        self.live.stop()

    def update_live(self):
        messages = sorted(self.messages.values(), key=lambda m: m.timestamp)
        content = []
        for message in messages:
            content.append(format_message(message, width=self.width))

        self.live.update(Group(*content), refresh=True)

    def on_message_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        self.messages[snapshot.id] = snapshot
        self.update_live()

    def on_tool_call_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        self.messages[snapshot.id] = snapshot
        self.update_live()

    def on_tool_result_done(self, message: ToolMessage):
        self.messages[message.id] = message
        self.update_live()
