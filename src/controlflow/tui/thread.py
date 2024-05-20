from marvin.beta.assistants.formatting import format_message, format_step
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


class Message(Static):
    message = reactive(None, recompose=True, always_update=True)

    def __init__(self, message, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self):
        yield Label(format_message(self.message))


class RunStep(Static):
    step = reactive(None, recompose=True, always_update=True)

    def __init__(self, step, **kwargs):
        super().__init__(**kwargs)
        self.step = step

    def compose(self):
        yield Label(format_step(self.step))
