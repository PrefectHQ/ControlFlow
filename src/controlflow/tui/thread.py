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
from controlflow.llm.messages import AssistantMessage, ToolMessage, UserMessage
from controlflow.llm.tools import get_tool_calls


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
        None, always_update=True, layout=True
    )

    def __init__(self, message: Union[UserMessage, AssistantMessage], **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def render(self):
        role_colors = {
            "user": "green",
            "assistant": "blue",
        }
        panels = []
        if tool_calls := get_tool_calls(self.message):
            for tool_call in tool_calls:
                content = Markdown(
                    inspect.cleandoc("""
                    :hammer_and_wrench: Calling `{name}` with the following arguments:
                    
                    ```json
                    {args}
                    ```
                    """).format(
                        name=tool_call.function.name, args=tool_call.function.arguments
                    )
                )
                panels.append(
                    Panel(
                        content,
                        title="[bold]Tool Call[/]",
                        subtitle=f"[italic]{format_timestamp(self.message.timestamp)}[/]",
                        title_align="left",
                        subtitle_align="right",
                        border_style=role_colors.get(self.message.role, "red"),
                        box=box.ROUNDED,
                        width=100,
                        expand=True,
                        padding=(1, 2),
                    )
                )
        else:
            role = {"assistant": "Agent", "user": "User"}

            panels.append(
                Panel(
                    Markdown(self.message.content or ""),
                    title=f"[bold]{role.get(self.message.role, 'Agent')}[/]",
                    subtitle=f"[italic]{format_timestamp(self.message.timestamp)}[/]",
                    title_align="left",
                    subtitle_align="right",
                    border_style=role_colors.get(self.message.role, "red"),
                    box=box.ROUNDED,
                    width=100,
                    expand=True,
                    padding=(1, 2),
                )
            )
        if len(panels) == 1:
            return panels[0]
        else:
            return Group(*panels)


class TUIToolMessage(Static):
    message: reactive[ToolMessage] = reactive(None, always_update=True, layout=True)

    def __init__(self, message: ToolMessage, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def render(self):
        if self.message.tool_metadata.get("is_failed"):
            content = f":x: The tool call to [markdown.code]{self.message.tool_call.function.name}[/] failed."
        elif not self.message.tool_metadata.get("is_task_status_tool"):
            content_type = (
                "json" if isinstance(self.message.tool_result, (dict, list)) else ""
            )
            content = Group(
                f":white_check_mark: Received output from the [markdown.code]{self.message.tool_call.function.name}[/] tool.\n",
                Markdown(f"```{content_type}\n{self.message.content or ''}\n```"),
            )
        else:
            self.display = False
            return ""

        return Panel(
            content,
            title="Tool Call Result",
            subtitle=f"[italic]{format_timestamp(self.message.timestamp)}[/]",
            title_align="left",
            subtitle_align="right",
            border_style="blue",
            box=box.ROUNDED,
            width=100,
            expand=True,
            padding=(1, 2),
        )
