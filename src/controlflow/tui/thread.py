import datetime
from typing import Literal

from marvin.beta.assistants.formatting import format_step
from rich import box
from rich.panel import Panel
from textual.reactive import reactive
from textual.widgets import Label, Static

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
    message = reactive(None, recompose=True, always_update=True)

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


class TUIRunStep(Static):
    step = reactive(None, recompose=True, always_update=True)

    def __init__(self, step, **kwargs):
        super().__init__(**kwargs)
        self.step = step

    def compose(self):
        yield Label(format_step(self.step))
