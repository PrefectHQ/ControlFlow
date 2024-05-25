from controlflow.llm.messages import AssistantMessage, ToolMessage
from controlflow.utilities.context import ctx


class CompletionHandler:
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
    def on_message_created(self, delta: AssistantMessage):
        print(f"Message created: {delta}\n")

    def on_message_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        print(f"Message delta: {delta}\n")

    def on_message_done(self, message: AssistantMessage):
        print(f"Message done: {message}\n")

    def on_tool_call_created(self, delta: AssistantMessage):
        print(f"Tool call created: {delta}\n")

    def on_tool_call_delta(self, delta: AssistantMessage, snapshot: AssistantMessage):
        print(f"Tool call delta: {delta}\n")

    def on_tool_call_done(self, message: AssistantMessage):
        print(f"Tool call done: {message}\n")

    def on_tool_result(self, message: ToolMessage):
        print(f"Tool result: {message}\n")
