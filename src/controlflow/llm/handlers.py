import rich
from rich.console import Group
from rich.live import Live

import controlflow
from controlflow.llm.formatting import format_message
from controlflow.llm.messages import (
    AIMessage,
    AIMessageChunk,
    MessageType,
    ToolCall,
    ToolMessage,
)
from controlflow.utilities.context import ctx
from controlflow.utilities.rich import console as cf_console
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
            "invalid_tool_call_done",
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

    def on_invalid_tool_call_done(self, message: AIMessage):
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


class PrintHandler(CompletionHandler):
    def __init__(self):
        self.width: int = controlflow.settings.print_handler_width
        self.messages: dict[str, MessageType] = {}
        self.live: Live = Live(auto_refresh=False, console=cf_console)
        self.paused_id: str = None
        super().__init__()

    def on_start(self):
        try:
            self.live.start()
        except rich.errors.LiveError:
            pass

    def on_end(self):
        self.live.stop()

    def on_exception(self, exc: Exception):
        self.live.stop()

    def update_live(self, latest: MessageType = None):
        # sort by timestamp, using the custom message id as a tiebreaker
        # in case the same message appears twice (e.g. tool call and message)
        messages = sorted(self.messages.items(), key=lambda m: (m[1].timestamp, m[0]))
        content = []
        for _, message in messages:
            content.append(format_message(message, width=self.width))

        if self.live.is_started:
            self.live.update(Group(*content), refresh=True)
        elif latest:
            cf_console.print(format_message(latest, width=self.width))

    def on_message_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        self.messages[snapshot.id] = snapshot
        self.update_live()

    def on_message_done(self, message: AIMessage):
        self.messages[message.id] = message
        self.update_live(latest=message)

    def on_tool_call_delta(self, delta: AIMessageChunk, snapshot: AIMessageChunk):
        self.messages[snapshot.id] = snapshot
        self.update_live()

    def on_tool_call_done(self, message: AIMessage):
        self.messages[message.id] = message
        self.update_live(latest=message)

    def on_tool_result_created(self, message: AIMessage, tool_call: ToolCall):
        # if collecting input on the terminal, pause the live display
        # to avoid overwriting the input prompt
        if tool_call["name"] == "talk_to_human":
            self.paused_id = tool_call["id"]
            self.live.stop()
            self.messages.clear()

    def on_tool_result_done(self, message: ToolMessage):
        self.messages[f"tool-result:{message.tool_call_id}"] = message

        # if we were paused, resume the live display
        if self.paused_id and self.paused_id == message.tool_call_id:
            self.paused_id = None
            # print newline to avoid odd formatting issues
            print()
            self.live = Live(auto_refresh=False)
            self.live.start()
        self.update_live(latest=message)
