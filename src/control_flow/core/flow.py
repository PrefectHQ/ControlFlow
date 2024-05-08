from typing import Callable, Literal

from marvin.beta.assistants import Thread
from openai.types.beta.threads import Message
from prefect import task as prefect_task
from pydantic import Field, field_validator

from control_flow.core.task import Task, TaskStatus
from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger
from control_flow.utilities.types import AssistantTool, ControlFlowModel

logger = get_logger(__name__)


class Flow(ControlFlowModel):
    thread: Thread = Field(None, validate_default=True)
    tools: list[AssistantTool | Callable] = Field(
        [], description="Tools that will be available to every agent in the flow"
    )
    model: str | None = None
    context: dict = {}
    tasks: dict[Task, int] = Field(repr=False, default_factory=dict)

    @field_validator("thread", mode="before")
    def _load_thread_from_ctx(cls, v):
        if v is None:
            v = ctx.get("thread", None)
            if v is None:
                v = Thread()
        if not v.id:
            v.create()

        return v

    def add_message(self, message: str, role: Literal["user", "assistant"] = None):
        prefect_task(self.thread.add)(message, role=role)

    def add_task(self, task: Task):
        if task not in self.tasks:
            task_id = len(self.tasks) + 1
            self.tasks[task] = task_id
            # this message is important for contexualizing activity
            # self.add_message(f'Task #{task_id} added to flow: "{task.objective}"')

    def get_task_id(self, task: Task):
        return self.tasks[task]

    def incomplete_tasks(self):
        return sorted(
            (t for t in self.tasks if t.status == TaskStatus.INCOMPLETE),
            key=lambda t: t.created_at,
        )

    def completed_tasks(self, reverse=False, limit=None):
        result = sorted(
            (t for t in self.tasks if t.status != TaskStatus.INCOMPLETE),
            key=lambda t: t.completed_at,
            reverse=reverse,
        )

        if limit:
            result = result[:limit]
        return result


def get_flow() -> Flow:
    """
    Loads the flow from the context.

    Will error if no flow is found in the context.
    """
    flow: Flow | None = ctx.get("flow")
    if not flow:
        raise ValueError("No flow found in context")
    return flow


def get_flow_messages(limit: int = None) -> list[Message]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    return flow.thread.get_messages(limit=limit)
