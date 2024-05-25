from rich.console import Group
from rich.live import Live

from controlflow.llm.formatting import format_message
from controlflow.llm.messages import (
    AssistantMessage,
    ControlFlowMessage,
    ToolMessage,
)
from controlflow.utilities.context import ctx


class CompletionHandler:
    def on_start(self):
        pass

    def on_end(self):
        pass

    def on_exception(self, exc: Exception):
        pass

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


class CompoundHandler(CompletionHandler):
    def __init__(self, handlers: list[CompletionHandler]):
        self.handlers = handlers

    def on_start(self):
        for handler in self.handlers:
            handler.on_start()

    def on_end(self):
        for handler in self.handlers:
            handler.on_end()

    def on_exception(self, exc: Exception):
        for handler in self.handlers:
            handler.on_exception(exc)

    def on_message_created(self, delta: AssistantMessage):
        for handler in self.handlers:
            handler.on_message_created(delta)

    def on_message_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        for handler in self.handlers:
            handler.on_message_delta(delta, snapshot)

    def on_message_done(self, message: AssistantMessage):
        for handler in self.handlers:
            handler.on_message_done(message)

    def on_tool_call_created(self, delta: AssistantMessage):
        for handler in self.handlers:
            handler.on_tool_call_created(delta)

    def on_tool_call_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        for handler in self.handlers:
            handler.on_tool_call_delta(delta, snapshot)

    def on_tool_call_done(self, message: AssistantMessage):
        for handler in self.handlers:
            handler.on_tool_call_done(message)

    def on_tool_result(self, message: ToolMessage):
        for handler in self.handlers:
            handler.on_tool_result(message)


class TUIHandler(CompletionHandler):
    def on_message_delta(
        self, delta: AssistantMessage, snapshot: AssistantMessage
    ) -> None:
        if tui := ctx.get("tui"):
            tui.update_message(message=snapshot)

    def on_tool_call_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        if tui := ctx.get("tui"):
            tui.update_message(message=snapshot)

    def on_tool_result(self, message: ToolMessage):
        if tui := ctx.get("tui"):
            tui.update_tool_result(message=message)


class PrintHandler(CompletionHandler):
    def on_start(self):
        self.live = Live(refresh_per_second=12)
        self.live.start()
        self.messages: dict[str, ControlFlowMessage] = {}

    def on_end(self):
        self.live.stop()

    def update_live(self):
        messages = sorted(self.messages.values(), key=lambda m: m.timestamp)
        content = []
        for message in messages:
            content.append(format_message(message))

        self.live.update(Group(*content))

    def on_message_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        self.messages[snapshot.id] = snapshot
        self.update_live()

    def on_tool_call_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        self.messages[snapshot.id] = snapshot
        self.update_live()

    def on_tool_result(self, message: ToolMessage):
        self.messages[message.id] = message
        self.update_live()
