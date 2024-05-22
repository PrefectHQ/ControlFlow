import datetime
import inspect
from typing import Union

from rich import box
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from textual.reactive import reactive
from textual.widgets import Static

from controlflow.core.task import TaskStatus
from controlflow.utilities.types import AssistantMessage, ToolMessage, UserMessage


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
    message: reactive[Union[UserMessage, AssistantMessage]] = reactive(
        None, always_update=True
    )

    def __init__(self, message: Union[UserMessage, AssistantMessage], **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def render(self):
        role_colors = {
            "user": "green",
            "assistant": "blue",
        }
        if isinstance(self.message, AssistantMessage) and self.message.has_tool_calls():
            content = Markdown(
                inspect.cleandoc("""
                :hammer_and_wrench: Calling `{name}` with the following arguments:
                
                ```json
                {args}
                ```
                """).format(name=self.tool_name, args=self.tool_args)
            )
            title = "Tool Call"
        else:
            content = self.message.content
            title = self.message.role.capitalize()
        return Panel(
            content,
            title=f"[bold]{title}[/]",
            subtitle=f"[italic]{format_timestamp(self.message.timestamp)}[/]",
            title_align="left",
            subtitle_align="right",
            border_style=role_colors.get(self.message.role, "red"),
            box=box.ROUNDED,
            width=100,
            expand=True,
            padding=(1, 2),
        )


class TUIToolMessage(Static):
    message: reactive[ToolMessage] = reactive(None, always_update=True)

    def __init__(self, message: ToolMessage, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def render(self):
        if self.message.tool_failed:
            content = f":x: The tool call to [markdown.code]{self.message.tool_name}[/] failed."
        else:
            content = Group(
                f":white_check_mark: Received output from the [markdown.code]{self.message.tool_call.function.name}[/] tool.\n",
                Markdown(f"```json\n{self.tool_result}\n```"),
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