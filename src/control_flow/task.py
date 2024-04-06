from enum import Enum
from typing import Generic, Optional, TypeVar

import marvin
import marvin.utilities.tools
from marvin.beta.assistants.runs import EndRun
from marvin.utilities.logging import get_logger
from marvin.utilities.tools import FunctionTool
from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class AITask(BaseModel, Generic[T]):
    """
    An AITask represents a single unit of work that an assistant must complete.
    Unlike instructions, which do not require a formal result and may last for
    zero or more agent iterations, tasks must be formally completed (or failed)
    by producing a result. Agents that are assigned tasks will continue to
    iterate until all tasks are completed.
    """

    objective: str
    instructions: Optional[str] = None
    context: dict = Field(None, validate_default=True)
    status: TaskStatus = TaskStatus.PENDING
    result: T = None
    error: Optional[str] = None

    # internal
    model_config: dict = dict(validate_assignment=True, extra="forbid")

    @field_validator("context", mode="before")
    def _default_context(cls, v):
        if v is None:
            v = {}
        return v

    def _create_complete_tool(
        self, task_id: int, end_run: bool = False
    ) -> FunctionTool:
        """
        Create an agent-compatible tool for completing this task.
        """

        result_type = self.get_result_type()

        if result_type is not None:

            def complete(result: result_type):
                self.result = result
                self.status = TaskStatus.COMPLETED
                if end_run:
                    return EndRun()

            tool = marvin.utilities.tools.tool_from_function(
                complete,
                name=f"complete_task_{task_id}",
                description=f"Mark task {task_id} completed",
            )
        else:

            def complete():
                self.status = TaskStatus.COMPLETED
                if end_run:
                    return EndRun()

            tool = marvin.utilities.tools.tool_from_function(
                complete,
                name=f"complete_task_{task_id}",
                description=f"Mark task {task_id} completed",
            )

        return tool

    def _create_fail_tool(self, task_id: int, end_run: bool = False) -> FunctionTool:
        """
        Create an agent-compatible tool for failing this task.
        """

        def fail(message: Optional[str] = None):
            self.error = message
            self.status = TaskStatus.FAILED
            if end_run:
                return EndRun()

        tool = marvin.utilities.tools.tool_from_function(
            fail,
            name=f"fail_task_{task_id}",
            description=f"Mark task {task_id} failed",
        )
        return tool

    def complete(self, result: T):
        self.result = result
        self.status = TaskStatus.COMPLETED

    def fail(self, message: Optional[str] = None):
        self.error = message
        self.status = TaskStatus.FAILED

    def get_result_type(self) -> T:
        """
        Returns the `type` of the task's result field.
        """
        return self.model_fields["result"].annotation
