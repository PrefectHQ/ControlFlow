from enum import Enum
from typing import Callable, Generic, TypeVar

import marvin
import marvin.utilities.tools
from marvin.utilities.logging import get_logger
from marvin.utilities.tools import FunctionTool
from pydantic import Field

from control_flow.types import AssistantTool, ControlFlowModel

T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(ControlFlowModel, Generic[T]):
    objective: str
    instructions: str | None = None
    context: dict = Field({})
    status: TaskStatus = TaskStatus.PENDING
    result: T = None
    error: str | None = None
    tools: list[AssistantTool | Callable] = []

    def __hash__(self):
        return id(self)

    def _create_complete_tool(self, task_id: int) -> FunctionTool:
        """
        Create an agent-compatible tool for completing this task.
        """
        result_type = self.get_result_type()

        def complete(result: result_type):
            self.result = result
            self.status = TaskStatus.COMPLETED

        tool = marvin.utilities.tools.tool_from_function(
            complete,
            name=f"complete_task_{task_id}",
            description=f"Mark task {task_id} completed",
        )

        return tool

    def _create_fail_tool(self, task_id: int) -> FunctionTool:
        """
        Create an agent-compatible tool for failing this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.fail,
            name=f"fail_task_{task_id}",
            description=f"Mark task {task_id} failed",
        )
        return tool

    def get_tools(self, task_id: int) -> list[AssistantTool | Callable]:
        return [
            self._create_complete_tool(task_id),
            self._create_fail_tool(task_id),
        ] + self.tools

    def complete(self, result: T):
        self.result = result
        self.status = TaskStatus.COMPLETED

    def fail(self, message: str | None = None):
        self.error = message
        self.status = TaskStatus.FAILED

    def get_result_type(self) -> T:
        """
        Returns the `type` of the task's result field.
        """
        return self.model_fields["result"].annotation
