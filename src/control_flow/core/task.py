import datetime
from enum import Enum
from typing import TYPE_CHECKING, Callable, TypeVar

import marvin
import marvin.utilities.tools
from marvin.utilities.tools import FunctionTool
from pydantic import Field, TypeAdapter

from control_flow.utilities.logging import get_logger
from control_flow.utilities.prefect import wrap_prefect_tool
from control_flow.utilities.types import AssistantTool, ControlFlowModel
from control_flow.utilities.user_access import talk_to_human

if TYPE_CHECKING:
    from control_flow.core.agent import Agent
T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    INCOMPLETE = "incomplete"
    SUCCESSFUL = "successful"
    FAILED = "failed"


class Task(ControlFlowModel):
    model_config = dict(extra="forbid", allow_arbitrary_types=True)
    objective: str
    instructions: str | None = None
    agents: list["Agent"] = []
    context: dict = {}
    status: TaskStatus = TaskStatus.INCOMPLETE
    result: T = None
    result_type: type[T] | None = str
    error: str | None = None
    tools: list[AssistantTool | Callable] = []
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    completed_at: datetime.datetime | None = None
    user_access: bool = False

    def __init__(self, objective, **kwargs):
        # allow objective as a positional arg
        super().__init__(objective=objective, **kwargs)

    def is_incomplete(self) -> bool:
        return self.status == TaskStatus.INCOMPLETE

    def is_complete(self) -> bool:
        return self.status != TaskStatus.INCOMPLETE

    def is_successful(self) -> bool:
        return self.status == TaskStatus.SUCCESSFUL

    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def __hash__(self):
        return id(self)

    def _create_success_tool(self, task_id: int) -> FunctionTool:
        """
        Create an agent-compatible tool for marking this task as successful.
        """

        # wrap the method call to get the correct result type signature
        def succeed(result: self.result_type):
            # validate the result
            self.mark_successful(result=result)

        tool = marvin.utilities.tools.tool_from_function(
            succeed,
            name=f"succeed_task_{task_id}",
            description=f"Mark task {task_id} as successful",
        )

        return tool

    def _create_fail_tool(self, task_id: int) -> FunctionTool:
        """
        Create an agent-compatible tool for failing this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_failed,
            name=f"fail_task_{task_id}",
            description=f"Mark task {task_id} as failed",
        )
        return tool

    def get_tools(self, task_id: int) -> list[AssistantTool | Callable]:
        tools = self.tools + [
            self._create_success_tool(task_id),
            self._create_fail_tool(task_id),
        ]
        if self.user_access:
            tools.append(marvin.utilities.tools.tool_from_function(talk_to_human))
        return [wrap_prefect_tool(t) for t in tools]

    def mark_successful(self, result: T = None):
        if self.result_type is None and result is not None:
            raise ValueError(
                f"Task {self.objective} specifies no result type, but a result was provided."
            )
        elif self.result_type is not None:
            result = TypeAdapter(self.result_type).validate_python(result)

        self.result = result
        self.status = TaskStatus.SUCCESSFUL
        self.completed_at = datetime.datetime.now()

    def mark_failed(self, message: str | None = None):
        self.error = message
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.datetime.now()


def any_incomplete(tasks: list[Task]) -> bool:
    return any(t.status == TaskStatus.INCOMPLETE for t in tasks)


def all_complete(tasks: list[Task]) -> bool:
    return all(t.status != TaskStatus.INCOMPLETE for t in tasks)


def all_successful(tasks: list[Task]) -> bool:
    return all(t.status == TaskStatus.SUCCESSFUL for t in tasks)


def any_failed(tasks: list[Task]) -> bool:
    return any(t.status == TaskStatus.FAILED for t in tasks)


def none_failed(tasks: list[Task]) -> bool:
    return not any_failed(tasks)
