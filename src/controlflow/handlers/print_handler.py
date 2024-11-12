import datetime
from typing import Optional, Union

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

# Global spinner for consistent animation
RUNNING_SPINNER = Spinner("dots")


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

    def get_status_style(self) -> tuple[Union[str, Spinner], str, str]:
        """Returns (icon, text style, border style) for current status."""
        if self.is_complete:
            if self.is_error:
                return "❌", "red", "red"
            else:
                return "✅", "green", "green3"  # Slightly softer green
        return (
            RUNNING_SPINNER,
            "yellow",
            "gray50",
        )  # Use shared spinner instance

    def render_completion_tool(
        self, show_inputs: bool = False, show_outputs: bool = False
    ) -> Panel:
        """Special rendering for completion tools."""
        table = Table.grid(padding=0, expand=True)
        header = Table.grid(padding=1)
        header.add_column(width=2)
        header.add_column()

        is_success_tool = self.tool.metadata.get("is_success_tool", False)
        is_fail_tool = self.tool.metadata.get("is_fail_tool", False)
        task = self.tool.metadata.get("completion_task")
        task_name = task.friendly_name() if task else "Unknown Task"
        # completion tools store their results on the task, rather than returning them directly
        task_result = task.result if task else None

        if not self.is_complete:
            icon = RUNNING_SPINNER  # Use shared spinner instance
            message = f"Working on task: {task_name}"
            text_style = "dim"
            border_style = "gray50"
        else:
            if self.is_error:
                icon = "❌"
                message = f"Error marking task status: {task_name}"
                text_style = "red"
                border_style = "red"
                if show_outputs and self.result:
                    message += f"\nError: {self.result}"
            elif is_fail_tool:
                icon = "❌"
                message = f"Task failed: {task_name}"
                text_style = "red"
                border_style = "red"
                if show_outputs and task_result:
                    message += f"\nReason: {task_result}"
            else:
                icon = "✓"
                message = f"Task complete: {task_name}"
                text_style = "dim"
                border_style = "gray50"

        header.add_row(icon, f"[{text_style}]{message}[/]")
        table.add_row(header)

        # Show details (streaming args or final result)
        if show_outputs and self.args:
            details = Table.grid(padding=(0, 2))
            details.add_column(style="dim", width=9)
            details.add_column()

            # If complete and successful, show task_result
            if (
                self.is_complete
                and not self.is_error
                and not is_fail_tool
                and task_result
            ):
                label = "Result" if is_success_tool else "Reason"
                details.add_row(
                    f"    {label}:",
                    f"{task_result}",
                )
            # Otherwise show streaming args
            else:
                label = "Result" if is_success_tool else "Reason"
                details.add_row(
                    f"    {label}:",
                    rich.pretty.Pretty(self.args, indent_size=2, expand_all=True),
                )
            table.add_row(details)

        return Panel(
            table,
            title=f"[bold]Agent: {self.agent_name}[/]",
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            title_align="left",
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            width=100,
            padding=(0, 1),
        )

    def render_panel(
        self,
        show_inputs: bool = True,
        show_outputs: bool = True,
    ) -> Panel:
        """Render tool state as a panel with status indicator."""
        if self.tool and self.tool.metadata.get("is_completion_tool"):
            return self.render_completion_tool(
                show_inputs=show_inputs, show_outputs=show_outputs
            )

        icon, text_style, border_style = self.get_status_style()
        table = Table.grid(padding=0, expand=True)

        header = Table.grid(padding=1)
        header.add_column(width=2)
        header.add_column()
        tool_name = self.name.replace("_", " ").title()
        header.add_row(icon, f"[{text_style} bold]{tool_name}[/]")
        table.add_row(header)

        if show_inputs or show_outputs:
            details = Table.grid(padding=(0, 2))
            details.add_column(style="dim", width=9)
            details.add_column()

            if show_inputs and self.args:
                details.add_row(
                    "    Input:",
                    rich.pretty.Pretty(self.args, indent_size=2, expand_all=True),
                )

            if show_outputs and self.is_complete and self.result:
                label = "Error" if self.is_error else "Output"
                style = "red" if self.is_error else "green3"
                details.add_row(
                    f"    {label}:",
                    f"[{style}]{self.result}[/]",
                )

            table.add_row(details)

        return Panel(
            table,
            title=f"[bold]Agent: {self.agent_name}[/]",
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            title_align="left",
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            width=100,
            padding=(0, 1),
        )


class PrintHandler(Handler):
    def __init__(
        self,
        show_completion_tools: bool = True,
        show_tool_inputs: bool = True,
        show_tool_outputs: bool = True,
        show_completion_tool_results: bool = False,
    ):
        super().__init__()
        # Tool display settings
        self.show_completion_tools = show_completion_tools
        self.show_tool_inputs = show_tool_inputs
        self.show_tool_outputs = show_tool_outputs
        # Completion tool specific settings
        self.show_completion_tool_results = show_completion_tool_results

        self.live: Optional[Live] = None
        self.paused_id: Optional[str] = None
        self.states: dict[str, DisplayState] = {}

    def update_display(self):
        """Render all current state as panels and update display."""
        if not self.live or not self.live.is_started or self.paused_id:
            return

        sorted_states = sorted(self.states.values(), key=lambda s: s.first_timestamp)
        panels = []

        for state in sorted_states:
            if isinstance(state, ToolState):
                is_completion_tool = state.tool and state.tool.metadata.get(
                    "is_completion_tool"
                )

                # Skip completion tools if disabled
                if not self.show_completion_tools and is_completion_tool:
                    continue

                if is_completion_tool:
                    panels.append(
                        state.render_completion_tool(
                            show_outputs=self.show_completion_tool_results
                        )
                    )
                else:
                    panels.append(
                        state.render_panel(
                            show_inputs=self.show_tool_inputs,
                            show_outputs=self.show_tool_outputs,
                        )
                    )
            else:
                panels.append(state.render_panel())

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

        tool_id = event.tool_call_snapshot["id"]
        if tool_id not in self.states:
            self.states[tool_id] = ToolState(
                agent_name=event.agent.name,
                first_timestamp=event.timestamp,
                name=event.tool_call_snapshot["name"],
                args=event.args,
                tool=event.tool,
            )
        else:
            state = self.states[tool_id]
            if isinstance(state, ToolState):
                state.args = event.args

        self.update_display()

    def on_tool_result(self, event: ToolResult):
        """Handle tool result events by updating tool state."""
        # Handle CLI input resume
        if event.tool_result.tool_call["name"] == "cli_input":
            if self.paused_id == event.tool_result.tool_call["id"]:
                self.paused_id = None
                print()
                self.live = Live(
                    console=cf_console,
                    vertical_overflow="visible",
                    auto_refresh=True,
                )
                self.live.start()
            return

        # Skip completion tools if disabled
        if (
            not self.show_completion_tools
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
            console=cf_console,
            vertical_overflow="visible",
            auto_refresh=True,
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
