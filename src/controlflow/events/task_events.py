from dataclasses import Field
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from pydantic.functional_serializers import PlainSerializer

from controlflow.events.base import UnpersistedEvent
from controlflow.tasks.task import Task


# Task events
class TaskStart(UnpersistedEvent):
    event: Literal["task-start"] = "task-start"
    task: Task


class TaskSuccess(UnpersistedEvent):
    event: Literal["task-success"] = "task-success"
    task: Task
    result: Annotated[
        Any,
        PlainSerializer(lambda x: str(x) if x else None, return_type=Optional[str]),
    ] = None


class TaskFailure(UnpersistedEvent):
    event: Literal["task-failure"] = "task-failure"
    task: Task
    reason: Optional[str] = None


class TaskSkipped(UnpersistedEvent):
    event: Literal["task-skipped"] = "task-skipped"
    task: Task
