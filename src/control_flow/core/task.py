import datetime
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    GenericAlias,
    Literal,
    TypeVar,
    _LiteralGenericAlias,
)

import marvin
import marvin.utilities.tools
from marvin.utilities.tools import FunctionTool
from pydantic import (
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from control_flow.core.flow import get_flow
from control_flow.instructions import get_instructions
from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger
from control_flow.utilities.prefect import wrap_prefect_tool
from control_flow.utilities.types import AssistantTool, ControlFlowModel
from control_flow.utilities.user_access import talk_to_human

if TYPE_CHECKING:
    from control_flow.core.agent import Agent
    from control_flow.core.graph import Graph
T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    INCOMPLETE = "incomplete"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    SKIPPED = "skipped"


NOTSET = "__notset__"


def visit_task_collection(
    val: Any, fn: Callable, recursion_limit: int = 3, _counter: int = 0
) -> list["Task"]:
    if _counter >= recursion_limit:
        return val

    if isinstance(val, dict):
        result = {}
        for key, value in list(val.items()):
            result[key] = visit_task_collection(
                value, fn=fn, recursion_limit=recursion_limit, _counter=_counter + 1
            )
    elif isinstance(val, (list, set, tuple)):
        result = []
        for item in val:
            result.append(
                visit_task_collection(
                    item, fn=fn, recursion_limit=recursion_limit, _counter=_counter + 1
                )
            )
    elif isinstance(val, Task):
        return fn(val)

    return val


class Task(ControlFlowModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:4]))
    objective: str = Field(
        ..., description="A brief description of the required result."
    )
    instructions: str | None = Field(
        None, description="Detailed instructions for completing the task."
    )
    agents: list["Agent"] = Field(None, validate_default=True)
    context: dict = Field(
        default_factory=dict,
        description="Additional context for the task. If tasks are provided as context, they are automatically added as `depends_on`",
    )
    subtasks: list["Task"] = Field(
        default_factory=list,
        description="A list of subtasks that are part of this task. Subtasks are considered dependencies, though they may be skipped.",
    )
    depends_on: list["Task"] = Field(
        default_factory=list, description="Tasks that this task depends on explicitly."
    )
    status: TaskStatus = TaskStatus.INCOMPLETE
    result: T = None
    result_type: type[T] | GenericAlias | _LiteralGenericAlias | None = None
    error: str | None = None
    tools: list[AssistantTool | Callable] = []
    user_access: bool = False
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    _parent: "Task | None" = None
    _downstreams: list["Task"] = []
    model_config = dict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(
        self, objective=None, result_type=None, parent: "Task" = None, **kwargs
    ):
        # allow certain args to be provided as a positional args
        if result_type is not None:
            kwargs["result_type"] = result_type
        if objective is not None:
            kwargs["objective"] = objective

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions", "")
                + "\n"
                + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)

        # setup up relationships
        if parent is None:
            parent_tasks = ctx.get("tasks", [])
            parent = parent_tasks[-1] if parent_tasks else None
        if parent is not None:
            parent.add_subtask(self)
        for task in self.depends_on:
            self.add_dependency(task)

    def __repr__(self):
        return str(self.model_dump())

    @field_validator("agents", mode="before")
    def _default_agent(cls, v):
        if v is None:
            v = get_flow().agents
        return v

    @field_validator("result_type", mode="before")
    def _turn_list_into_literal_result_type(cls, v):
        if isinstance(v, (list, tuple, set)):
            return Literal[tuple(v)]  # type: ignore
        return v

    @model_validator(mode="after")
    def _load_context_dependencies(self):
        tasks = []

        def visitor(task):
            tasks.append(task)
            return task

        visit_task_collection(self.context, visitor)
        for task in tasks:
            if task not in self.depends_on:
                self.depends_on.append(task)
        return self

    @field_serializer("subtasks")
    def _serialize_subtasks(subtasks: list["Task"]):
        return [t.id for t in subtasks]

    @field_serializer("depends_on")
    def _serialize_depends_on(depends_on: list["Task"]):
        return [t.id for t in depends_on]

    @field_serializer("context")
    def _serialize_context(context: dict):
        def visitor(task):
            return f"<Result from task {task.id}>"

        return visit_task_collection(context, visitor)

    @field_serializer("result_type")
    def _serialize_result_type(result_type: list["Task"]):
        return repr(result_type)

    @field_serializer("agents")
    def _serialize_agents(agents: list["Agent"]):
        return [
            a.model_dump(include={"name", "description", "tools", "user_access"})
            for a in agents
        ]

    def as_graph(self) -> "Graph":
        from control_flow.core.graph import Graph

        return Graph.from_tasks(tasks=[self])

    def add_subtask(self, task: "Task"):
        """
        Indicate that this task has a subtask (which becomes an implicit dependency).
        """
        if task._parent is None:
            task._parent = self
        elif task._parent is not self:
            raise ValueError(f"Task {task.id} already has a parent.")
        if task not in self.subtasks:
            self.subtasks.append(task)

    def add_dependency(self, task: "Task"):
        """
        Indicate that this task depends on another task.
        """
        if task not in self.depends_on:
            self.depends_on.append(task)
        if self not in task._downstreams:
            task._downstreams.append(self)

    def run_once(self, agent: "Agent" = None, run_dependencies: bool = True):
        """
        Runs the task with provided agent. If no agent is provided, one will be selected from the task's agents.
        """
        from control_flow.core.controller import Controller

        controller = Controller(
            tasks=[self], agents=agent, run_dependencies=run_dependencies
        )

        controller.run_once()

    def run(self, run_dependencies: bool = True) -> T:
        """
        Runs the task with provided agents until it is complete.
        """
        while self.is_incomplete():
            self.run_once(run_dependencies=run_dependencies)
            if self.is_successful():
                return self.result
            elif self.is_failed():
                raise ValueError(f"Task {self.id} failed: {self.error}")

    @contextmanager
    def _context(self):
        stack = ctx.get("tasks", [])
        stack.append(self)
        with ctx(tasks=stack):
            yield self

    def __enter__(self):
        self.__cm = self._context()
        return self.__cm.__enter__()

    def __exit__(self, *exc_info):
        return self.__cm.__exit__(*exc_info)

    def is_incomplete(self) -> bool:
        return self.status == TaskStatus.INCOMPLETE

    def is_complete(self) -> bool:
        return self.status != TaskStatus.INCOMPLETE

    def is_successful(self) -> bool:
        return self.status == TaskStatus.SUCCESSFUL

    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def is_skipped(self) -> bool:
        return self.status == TaskStatus.SKIPPED

    def is_ready(self) -> bool:
        """
        Returns True if all dependencies are complete and this task is incomplete.
        """
        return self.is_incomplete() and all(t.is_complete() for t in self.depends_on)

    def __hash__(self):
        return id(self)

    def _create_success_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for marking this task as successful.
        """

        # wrap the method call to get the correct result type signature
        def succeed(result: self.result_type) -> str:
            return self.mark_successful(result=result)

        tool = marvin.utilities.tools.tool_from_function(
            succeed,
            name=f"mark_task_{self.id}_successful",
            description=f"Mark task {self.id} as successful and optionally provide a result.",
        )

        return tool

    def _create_fail_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for failing this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_failed,
            name=f"mark_task_{self.id}_failed",
            description=f"Mark task {self.id} as failed. Only use when a technical issue like a broken tool or unresponsive human prevents completion.",
        )
        return tool

    def _create_skip_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for skipping this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_skipped,
            name=f"mark_task_{self.id}_skipped",
            description=f"Mark task {self.id} as skipped. Only use when completing its parent task early.",
        )
        return tool

    def get_tools(self, validate: bool = True) -> list[AssistantTool | Callable]:
        tools = self.tools.copy()
        if self.is_incomplete():
            tools.extend([self._create_fail_tool(), self._create_success_tool()])
            # add skip tool if this task has a parent task
            if self._parent is not None:
                tools.append(self._create_skip_tool())
        if self.user_access:
            tools.append(marvin.utilities.tools.tool_from_function(talk_to_human))
        return [wrap_prefect_tool(t) for t in tools]

    def mark_successful(self, result: T = None, validate: bool = True):
        if validate:
            if any(t.is_incomplete() for t in self.depends_on):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "upstream dependencies are completed. Incomplete dependencies "
                    f"are: {[t.id for t in self.depends_on if t.is_incomplete()]}"
                )
            elif any(t.is_incomplete() for t in self.subtasks):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "subtasks are completed. Incomplete subtasks "
                    f"are: {[t.id for t in self.subtasks if t.is_incomplete()]}"
                )

        if self.result_type is None and result is not None:
            raise ValueError(
                f"Task {self.objective} specifies no result type, but a result was provided."
            )
        elif self.result_type is not None:
            result = TypeAdapter(self.result_type).validate_python(result)

        self.result = result
        self.status = TaskStatus.SUCCESSFUL
        return f"Task {self.id} marked successful. Updated task definition: {self.model_dump()}"

    def mark_failed(self, message: str | None = None):
        self.error = message
        self.status = TaskStatus.FAILED
        return f"Task {self.id} marked failed. Updated task definition: {self.model_dump()}"

    def mark_skipped(self):
        self.status = TaskStatus.SKIPPED
        return f"Task {self.id} marked skipped. Updated task definition: {self.model_dump()}"


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
