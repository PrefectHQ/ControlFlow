from typing import Literal

from controlflow.events.events import Event, UnpersistedEvent
from controlflow.tasks.task import Task
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


class TaskReadyEvent(UnpersistedEvent):
    event: Literal["task-ready"] = "task-ready"
    task: Task


class TaskCompleteEvent(Event):
    event: Literal["task-complete"] = "task-complete"
    task: Task

    # def to_messages(self, context: EventContext) -> list[BaseMessage]:
    #     return [
    #         SystemMessage(
    #             content=f"Task {self.task.id} is complete with status: {self.task.status}"
    #         )
    #     ]
