from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Collapsible, Label, Static

from controlflow.core.task import Task

from .basic import ReactiveLabel, Row


def bool_to_emoji(value: bool) -> str:
    return "✅" if value else "❌"


class EmojiStatus(Label):
    status: str = reactive(None)

    def render(self) -> str:
        if self.status == "READY":
            return "🔜"
        elif self.status == "INCOMPLETE":
            return "⏳"
        elif self.status == "SUCCESSFUL":
            return "✅"
        elif self.status == "FAILED":
            return "❌"
        elif self.status == "SKIPPED":
            return "⏭️"
        else:
            return "❓"


class TaskResult(Label):
    result: str = reactive(None)

    def render(self) -> str:
        return str(self.result)


class TUITask(Static):
    task: Task = reactive(None, always_update=True)
    status: str = reactive(None)
    result: str = reactive(None)
    error_msg: str = reactive(None)

    def __init__(self, task: Task, **kwargs):
        super().__init__(**kwargs)
        self.task = task

    def on_mount(self):
        def refresh():
            self.task = self.task

        # refresh the task periodically in case it goes "ready"
        self.set_interval(1 / 2, refresh)

    def watch_task(self, task: Task):
        if task is None:
            return

        if task.is_ready():
            self.status = "READY"
        else:
            self.status = task.status.value

        self.result = task.result
        self.error_msg = task.error
        if self.is_mounted:
            if self.result is not None:
                self.query_one(".result-collapsible", Collapsible).display = "block"
            if self.error_msg is not None:
                self.query_one(".error-collapsible", Collapsible).display = "block"

    def compose(self):
        self.border_title = f"Task {self.task.id}"
        with Row(classes="task-status-row"):
            yield (
                EmojiStatus(classes="status task-info").data_bind(status=TUITask.status)
            )
            yield Label(self.task.objective, classes="objective task-info")

        with Vertical(classes="task-info-row"):
            yield Label(
                f"Agents: {', '.join(a.name for a in self.task.get_agents())}",
                classes="user-access task-info",
            )
            # yield Label(
            #     f"User access: {self.task.user_access}",
            #     classes="user-access task-info",
            # )

            # ------------------ success

            result_collapsible = Collapsible(
                title="Result", classes="task-info result-collapsible"
            )
            result_collapsible.display = "none"
            with result_collapsible:
                yield ReactiveLabel(classes="task-info result").data_bind(
                    value=TUITask.result
                )

            # ------------------ failure

            error_collapsible = Collapsible(
                title="Error", classes="task-info error-collapsible"
            )
            error_collapsible.display = "none"
            with error_collapsible:
                yield ReactiveLabel(classes="task-info error").data_bind(
                    value=TUITask.error_msg
                )
