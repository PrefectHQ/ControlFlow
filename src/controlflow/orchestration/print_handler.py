import datetime
from typing import Union

import rich
from rich import box
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

import controlflow
from controlflow.events.base import Event
from controlflow.events.events import (
    AgentMessage,
    AgentMessageDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from controlflow.events.orchestrator_events import (
    OrchestratorEnd,
    OrchestratorError,
    OrchestratorStart,
)
from controlflow.llm.messages import BaseMessage
from controlflow.orchestration.handler import Handler
from controlflow.tools.tools import ToolCall
from controlflow.utilities.rich import console as cf_console


class PrintHandler(Handler):
    def __init__(self):
        self.events: dict[str, Event] = {}
        self.paused_id: str = None
        super().__init__()

    def update_live(self, latest: BaseMessage = None):
        events = sorted(self.events.items(), key=lambda e: (e[1].timestamp, e[0]))
        content = []

        tool_results = {}  # To track tool results by their call ID

        # gather all tool events first
        for _, event in events:
            if isinstance(event, ToolResultEvent):
                tool_results[event.tool_call["id"]] = event

        for _, event in events:
            if isinstance(event, (AgentMessageDelta, AgentMessage)):
                if formatted := format_event(event, tool_results=tool_results):
                    content.append(formatted)

        if not content:
            return
        elif self.live.is_started:
            self.live.update(Group(*content), refresh=True)
        elif latest:
            cf_console.print(format_event(latest))

    def on_orchestrator_start(self, event: OrchestratorStart):
        self.live: Live = Live(
            auto_refresh=False, console=cf_console, vertical_overflow="visible"
        )
        self.events.clear()
        try:
            self.live.start()
        except rich.errors.LiveError:
            pass

    def on_orchestrator_end(self, event: OrchestratorEnd):
        self.live.stop()

    def on_orchestrator_error(self, event: OrchestratorError):
        self.live.stop()

    def on_agent_message_delta(self, event: AgentMessageDelta):
        self.events[event.snapshot_message.id] = event
        self.update_live()

    def on_agent_message(self, event: AgentMessage):
        self.events[event.ai_message.id] = event
        self.update_live()

    def on_tool_call(self, event: ToolCallEvent):
        # if collecting input on the terminal, pause the live display
        # to avoid overwriting the input prompt
        if event.tool_call["name"] == "cli_input":
            self.paused_id = event.tool_call["id"]
            self.live.stop()
            self.events.clear()

    def on_tool_result(self, event: ToolResultEvent):
        self.events[f"tool-result:{event.tool_call['id']}"] = event

        # # if we were paused, resume the live display
        if self.paused_id and self.paused_id == event.tool_call["id"]:
            self.paused_id = None
            # print newline to avoid odd formatting issues
            print()
            self.live = Live(auto_refresh=False)
            self.live.start()
        self.update_live(latest=event)


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


def format_timestamp(timestamp: datetime.datetime) -> str:
    local_timestamp = timestamp.astimezone()
    return local_timestamp.strftime("%I:%M:%S %p").lstrip("0").rjust(11)


def status(icon, text) -> Table:
    t = Table.grid(padding=1)
    t.add_row(icon, text)
    return t


def format_event(
    event: Union[AgentMessageDelta, AgentMessage],
    tool_results: dict[str, ToolResultEvent] = None,
) -> Panel:
    title = f"Agent: {event.agent.name}"

    content = []
    if isinstance(event, AgentMessageDelta):
        message = event.snapshot_message
    elif isinstance(event, AgentMessage):
        message = event.ai_message
    else:
        return

    if message.content:
        if isinstance(message.content, str):
            content.append(Markdown(str(message.content)))
        elif isinstance(message.content, dict):
            if "content" in message.content:
                content.append(Markdown(str(message.content["content"])))
            elif "text" in message.content:
                content.append(Markdown(str(message.content["text"])))
        elif isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, str):
                    content.append(Markdown(str(item)))
                elif "content" in item:
                    content.append(Markdown(str(item["content"])))
                elif "text" in item:
                    content.append(Markdown(str(item["text"])))

    tool_content = []
    for tool_call in message.tool_calls + message.invalid_tool_calls:
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
        subtitle=f"[italic]{format_timestamp(event.timestamp)}[/]",
        title_align="left",
        subtitle_align="right",
        border_style=ROLE_COLORS.get("ai", "red"),
        box=box.ROUNDED,
        width=100,
        expand=True,
        padding=(1, 2),
    )


def format_tool_call(tool_call: ToolCall) -> Panel:
    if controlflow.settings.tools_verbose:
        return status(
            Spinner("dots"),
            f'Tool call: "{tool_call["name"]}"\n\nTool args: {tool_call["args"]}',
        )
    return status(Spinner("dots"), f'Tool call: "{tool_call["name"]}"')


def format_tool_result(event: ToolResultEvent) -> Panel:
    if event.tool_result.is_error:
        icon = ":x:"
    else:
        icon = ":white_check_mark:"

    if controlflow.settings.tools_verbose:
        msg = f'Tool call: "{event.tool_call["name"]}"\n\nTool args: {event.tool_call["args"]}\n\nTool result: {event.tool_result.str_result}'
    else:
        msg = f'Tool call: "{event.tool_call["name"]}"'
    return status(icon, msg)
