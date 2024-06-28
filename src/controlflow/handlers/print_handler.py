import datetime

import rich
from rich import box
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

import controlflow
from controlflow.llm.handlers import CompletionHandler
from controlflow.llm.messages import (
    AIMessage,
    AIMessageChunk,
    MessageType,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from controlflow.utilities.rich import console as cf_console


class PrintHandler(CompletionHandler):
    def __init__(self):
        self.messages: dict[str, MessageType] = {}
        self.live: Live = Live(auto_refresh=True, console=cf_console)
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

        tool_results = {}  # To track tool results by their call ID

        # gather all tool messages first
        for _, message in messages:
            if isinstance(message, ToolMessage):
                tool_results[message.tool_call_id] = message

        for _, message in messages:
            if isinstance(message, (SystemMessage, UserMessage, AIMessage)):
                content.append(format_message(message, tool_results=tool_results))
            # no need to handle tool messages

        if self.live.is_started:
            self.live.update(Group(*content), refresh=True)
        elif latest:
            cf_console.print(format_message(latest))

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
        if tool_call["name"] == "talk_to_user":
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


def format_timestamp(timestamp: datetime.datetime) -> str:
    local_timestamp = timestamp.astimezone()
    return local_timestamp.strftime("%I:%M:%S %p").lstrip("0").rjust(11)


def status(icon, text) -> Table:
    t = Table.grid(padding=1)
    t.add_row(icon, text)
    return t


ROLE_COLORS = {
    "system": "gray",
    "ai": "blue",
    "user": "green",
}
ROLE_NAMES = {
    "system": "System",
    "ai": "Agent",
    "user": "User",
}


def format_message(message: MessageType, tool_results: dict = None) -> Panel:
    if message.role == "ai" and message.name:
        title = f"Agent: {message.name}"
    else:
        title = ROLE_NAMES.get(message.role, "Agent")

    content = []
    if message.str_content:
        content.append(Markdown(message.str_content or ""))

    tool_content = []
    for tool_call in getattr(message, "tool_calls", []):
        tool_result = (tool_results or {}).get(tool_call["id"])
        if tool_result:
            c = format_tool_result(tool_result)

        else:
            c = format_tool_call(tool_call)
        if c:
            tool_content.append(c)

    if content and tool_content:
        content.append("\n")

    return Panel(
        Group(*content, *tool_content),
        title=f"[bold]{title}[/]",
        subtitle=f"[italic]{format_timestamp(message.timestamp)}[/]",
        title_align="left",
        subtitle_align="right",
        border_style=ROLE_COLORS.get(message.role, "red"),
        box=box.ROUNDED,
        width=100,
        expand=True,
        padding=(1, 2),
    )


def format_tool_call(tool_call: ToolCall) -> Panel:
    name = tool_call["name"]
    args = tool_call["args"]
    if controlflow.settings.tools_verbose:
        return status(Spinner("dots"), f'Tool call: "{name}"\n\nTool args: {args}')
    return status(Spinner("dots"), f'Tool call: "{name}"')


def format_tool_result(message: ToolMessage) -> Panel:
    name = message.tool_call["name"]

    if message.is_error:
        icon = ":x:"
    else:
        icon = ":white_check_mark:"

    if controlflow.settings.tools_verbose:
        msg = f'Tool call: "{name}"\n\nTool result: {message.str_content}'
    else:
        msg = f'Tool call: "{name}"'
    return status(icon, msg)
