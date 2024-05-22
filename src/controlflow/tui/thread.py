import datetime
import inspect
from typing import Literal

from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from textual.reactive import reactive
from textual.widgets import Static

from controlflow.core.task import TaskStatus


def bool_to_emoji(value: bool) -> str:
    return "✅" if value else "❌"


def status_to_emoji(status: TaskStatus) -> str:
    if status == TaskStatus.INCOMPLETE:
        return "🔄"
    elif status == TaskStatus.SUCCESSFUL:
        return "✅"
    elif status == TaskStatus.FAILED:
        return "❌"
    elif status == TaskStatus.SKIPPED:
        return "⏭️"
    else:
        return "❓"


def format_timestamp(timestamp: datetime.datetime) -> str:
    return timestamp.strftime("%l:%M:%S %p")


class TUIMessage(Static):
    message: reactive[str] = reactive(None, always_update=True)

    def __init__(
        self,
        message: str,
        role: Literal["user", "assistant"] = "assistant",
        timestamp: datetime.datetime = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if timestamp is None:
            timestamp = datetime.datetime.now()
        self._timestamp = timestamp
        self._role = role
        self.message = message

    def render(self):
        role_colors = {
            "user": "green",
            "assistant": "blue",
        }
        return Panel(
            self.message,
            title=f"[bold]{self._role.capitalize()}[/]",
            subtitle=f"[italic]{format_timestamp(self._timestamp)}[/]",
            title_align="left",
            subtitle_align="right",
            border_style=role_colors.get(self._role, "red"),
            box=box.ROUNDED,
            width=100,
            expand=True,
            padding=(1, 2),
        )


class TUIToolCall(Static):
    tool_name: reactive[str] = reactive(None, always_update=True)
    tool_args: reactive[str] = reactive(None, always_update=True)

    def __init__(
        self,
        tool_name: str,
        tool_args: str,
        timestamp: datetime.datetime = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if timestamp is None:
            timestamp = datetime.datetime.now()
        self._timestamp = timestamp
        self.tool_name = tool_name
        self.tool_args = tool_args

    def render(self):
        content = inspect.cleandoc("""
            :hammer_and_wrench: Calling `{name}` with the following arguments:
            
            ```json
            {args}
            ```
            """).format(name=self.tool_name, args=self.tool_args)
        return Panel(
            Markdown(content),
            title="Tool Call",
            subtitle=f"[italic]{format_timestamp(self._timestamp)}[/]",
            title_align="left",
            subtitle_align="right",
            border_style="blue",
            box=box.ROUNDED,
            width=100,
            expand=True,
            padding=(1, 2),
        )


class TUIToolResult(Static):
    tool_name: reactive[str] = reactive(None, always_update=True)
    tool_result: reactive[str] = reactive(None, always_update=True)

    def __init__(
        self,
        tool_name: str,
        tool_result: str,
        timestamp: datetime.datetime = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if timestamp is None:
            timestamp = datetime.datetime.now()
        self._timestamp = timestamp
        self.tool_name = tool_name
        self.tool_result = tool_result

    def render(self):
        content = Markdown(
            f":white_check_mark: Received output from the [markdown.code]{self.tool_name}[/] tool."
            f"\n\n```json\n{self.tool_result}\n```",
        )

        return Panel(
            content,
            title="Tool Call Result",
            subtitle=f"[italic]{format_timestamp(self._timestamp)}[/]",
            title_align="left",
            subtitle_align="right",
            border_style="blue",
            box=box.ROUNDED,
            width=100,
            expand=True,
            padding=(1, 2),
        )
