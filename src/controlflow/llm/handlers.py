from controlflow.llm.messages import (
    AIMessage,
    AIMessageChunk,
    MessageType,
    ToolCall,
    ToolMessage,
)
from controlflow.utilities.context import ctx
from controlflow.utilities.types import ControlFlowModel


class CompletionEvent(ControlFlowModel):
    type: str
    payload: dict


class CompletionHandler:
    def __init__(self):
        self._response_message_ids: set[str] = set()

    def on_event(self, event: CompletionEvent):
        method = getattr(self, f"on_{event.type}", None)
        if not method:
            raise ValueError(f"Unknown event type: {event.type}")
        method(**event.payload)
        if event.type in [
            "message_done",
            "tool_call_done",
            "tool_result_done",
        ]:
            # only fire the on_response_message hook once per message
            # (a message could contain both a tool call and a message)
            if event.payload["message"].id not in self._response_message_ids:
                self.on_response_message(event.payload["message"])
            self._response_message_ids.add(event.payload["message"].id)

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

    def on_tool_result_created(self, message: AIMessage, tool_call: ToolCall):
        pass

    def on_tool_result_done(self, message: ToolMessage):
        pass

    def on_response_message(self, message: MessageType):
        """
        This handler is called whenever a message is generated that should be
        included in the completion history (e.g. a `message`, `tool_call` or
        `tool_result`). Note that this is called *in addition* to the respective
        on_*_done handlers, and can be used to quickly collect all messages
        generated during a completion. Messages that satisfy multiple criteria
        (e.g. a message and a tool call) will only be included once.
        """
        pass


class ResponseHandler(CompletionHandler):
    """
    A handler for collecting response messages.
    """

    def __init__(self):
        super().__init__()
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
