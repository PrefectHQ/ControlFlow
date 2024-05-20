from textual.reactive import reactive
from textual.widgets import Label, Rule, Static

from controlflow.core.task import Task as CFTask
from controlflow.core.task import TaskStatus

from .basic import Row


def bool_to_emoji(value: bool) -> str:
    return "✅" if value else "❌"


def status_to_emoji(task: CFTask) -> str:
    if task.is_ready():
        return "🔜"
    if task.status == TaskStatus.INCOMPLETE:
        return "⏳"
    elif task.status == TaskStatus.SUCCESSFUL:
        return "✅"
    elif task.status == TaskStatus.FAILED:
        return "❌"
    elif task.status == TaskStatus.SKIPPED:
        return "⏭️"
    else:
        return "❓"


class Task(Static):
    task: CFTask = reactive(None, recompose=True, always_update=True)

    def __init__(self, task: CFTask, **kwargs):
        super().__init__(**kwargs)
        self.task = task

    def on_mount(self):
        def refresh():
            self.task = self.task

        # refresh the task periodically in case it goes "ready"
        self.set_interval(1 / 2, refresh)

    def compose(self):
        with Row(classes="task-status-row"):
            yield Label(status_to_emoji(self.task), classes="status task-info")
            yield Label(self.task.objective, classes="objective task-info")

        with Row(classes="task-info-row"):
            yield Label(
                f"ID: {self.task.id}",
                classes="task-info",
            )
            yield Rule(orientation="vertical")
            yield Label(
                f"Agents: {', '.join(a.name for a in self.task.get_agents())}",
                classes="user-access task-info",
            )
            yield Rule(orientation="vertical")
            yield Label(
                f"User access: {self.task.user_access}",
                classes="user-access task-info",
            )
