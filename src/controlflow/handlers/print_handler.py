import datetime
from typing import Optional

import rich
from pydantic import BaseModel
from rich import box
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table

from controlflow.events.events import AgentContentDelta, AgentToolCallDelta, ToolResult
from controlflow.events.orchestrator_events import (
    OrchestratorEnd,
    OrchestratorError,
    OrchestratorStart,
)
from controlflow.orchestration.handler import Handler
from controlflow.tools.tools import Tool
from controlflow.utilities.rich import console as cf_console


class DisplayState(BaseModel):
    """Base class for content to be displayed."""

    agent_name: str
    first_timestamp: datetime.datetime

    def format_timestamp(self) -> str:
        """Format the timestamp for display."""
        local_timestamp = self.first_timestamp.astimezone()
        return local_timestamp.strftime("%I:%M:%S %p").lstrip("0").rjust(11)


class ContentState(DisplayState):
    """State for content being streamed."""

    content: str = ""

    @staticmethod
    def _convert_content_to_str(content) -> str:
        """Convert various content formats to a string."""
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            return content.get("content", content.get("text", ""))

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    part = item.get("content", item.get("text", ""))
                    if part:
                        parts.append(part)
            return "\n".join(parts)

        return str(content)

    def update_content(self, new_content) -> None:
        """Update content, converting complex content types to string."""
        self.content = self._convert_content_to_str(new_content)

    def render_panel(self) -> Panel:
        """Render content as a markdown panel."""
        return Panel(
            Markdown(self.content),
            title=f"[bold]Agent: {self.agent_name}[/]",
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            title_align="left",
            subtitle_align="right",
            border_style="blue",
            box=box.ROUNDED,
            width=100,
            padding=(1, 2),
        )


class ToolState(DisplayState):
    """State for a tool call and its result."""

    name: str
    args: dict
    result: Optional[str] = None
    is_error: bool = False
    is_complete: bool = False
    tool: Optional[Tool] = None

    def render_panel(self, show_details: bool = True) -> Panel:
        """Render tool state as a panel with status indicator."""
        t = Table.grid(padding=1)

        if self.is_complete:
            icon = ":x:" if self.is_error else ":white_check_mark:"
            if show_details and self.result:
                tool_text = f'Tool "{self.name}": {self.result}'
            else:
                tool_text = f'Tool "{self.name}" completed'
        else:
            icon = Spinner("dots")
            tool_text = f'Tool "{self.name}" running...'
            if show_details and self.args:
                tool_text += f"\nArguments: {self.args}"

        t.add_row(icon, tool_text)

        return Panel(
            t,
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            subtitle_align="right",
            border_style="red" if self.is_error else "blue",
            box=box.ROUNDED,
            width=100,
            padding=(1, 2),
        )


class PrintHandler(Handler):
    def __init__(self, include_completion_tools: bool = True):
        super().__init__()
        self.include_completion_tools = include_completion_tools
        self.live: Optional[Live] = None
        self.paused_id: Optional[str] = None
        self.states: dict[str, DisplayState] = {}

    def update_display(self):
        """Render all current state as panels and update display."""
        if not self.live or not self.live.is_started or self.paused_id:
            return

        # Sort states by timestamp and render panels
        sorted_states = sorted(self.states.values(), key=lambda s: s.first_timestamp)
        panels = [
            state.render_panel(show_details=self.include_completion_tools)
            if isinstance(state, ToolState)
            else state.render_panel()
            for state in sorted_states
        ]

        if panels:
            self.live.update(Group(*panels), refresh=True)

    def on_agent_content_delta(self, event: AgentContentDelta):
        """Handle content delta events by updating content state."""
        if not event.content_delta:
            return
        if event.agent_message_id not in self.states:
            state = ContentState(
                agent_name=event.agent.name,
                first_timestamp=event.timestamp,
            )
            state.update_content(event.content_snapshot)
            self.states[event.agent_message_id] = state
        else:
            state = self.states[event.agent_message_id]
            if isinstance(state, ContentState):
                state.update_content(event.content_snapshot)

        self.update_display()

    def on_agent_tool_call_delta(self, event: AgentToolCallDelta):
        """Handle tool call delta events by updating tool state."""
        # Handle CLI input special case
        if event.tool_call_snapshot["name"] == "cli_input":
            self.paused_id = event.tool_call_snapshot["id"]
            if self.live and self.live.is_started:
                self.live.stop()
            return

        # Skip completion tools if configured
        if (
            not self.include_completion_tools
            and event.tool
            and event.tool.metadata.get("is_completion_tool")
        ):
            return

        tool_id = event.tool_call_snapshot["id"]
        if tool_id not in self.states:
            self.states[tool_id] = ToolState(
                agent_name=event.agent.name,
                first_timestamp=event.timestamp,
                name=event.tool_call_snapshot["name"],
                args=event.args,
                tool=event.tool,
            )

        self.update_display()

    def on_tool_result(self, event: ToolResult):
        """Handle tool result events by updating tool state."""
        # Handle CLI input resume
        if event.tool_result.tool_call["name"] == "cli_input":
            if self.paused_id == event.tool_result.tool_call["id"]:
                self.paused_id = None
                print()
                self.live = Live(console=cf_console, auto_refresh=False)
                self.live.start()
            return

        # Skip completion tools if configured
        if (
            not self.include_completion_tools
            and event.tool_result.tool
            and event.tool_result.tool.metadata.get("is_completion_tool")
        ):
            return

        tool_id = event.tool_result.tool_call["id"]
        if tool_id in self.states:
            state = self.states[tool_id]
            if isinstance(state, ToolState):
                state.is_complete = True
                state.is_error = event.tool_result.is_error
                state.result = event.tool_result.str_result

        self.update_display()

    def on_orchestrator_start(self, event: OrchestratorStart):
        """Initialize live display."""
        self.live = Live(
            auto_refresh=False, console=cf_console, vertical_overflow="visible"
        )
        self.states.clear()
        try:
            self.live.start()
        except rich.errors.LiveError:
            pass

    def on_orchestrator_end(self, event: OrchestratorEnd):
        """Clean up live display."""
        if self.live and self.live.is_started:
            self.live.stop()

    def on_orchestrator_error(self, event: OrchestratorError):
        """Clean up live display on error."""
        if self.live and self.live.is_started:
            self.live.stop()
