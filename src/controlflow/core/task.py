import datetime
import functools
import inspect
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    GenericAlias,
    Literal,
    TypeVar,
    Union,
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

import controlflow
from controlflow.instructions import get_instructions
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import wrap_prefect_tool
from controlflow.utilities.tasks import collect_tasks, visit_task_collection
from controlflow.utilities.types import (
    NOTSET,
    AssistantTool,
    ControlFlowModel,
    ToolType,
)
from controlflow.utilities.user_access import talk_to_human

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.graph import Graph
T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    INCOMPLETE = "incomplete"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(ControlFlowModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:5]))
    objective: str = Field(
        ..., description="A brief description of the required result."
    )
    instructions: Union[str, None] = Field(
        None, description="Detailed instructions for completing the task."
    )
    agents: Union[list["Agent"], None] = Field(
        None,
        description="The agents assigned to the task. If None, the task will use its flow's default agents.",
        validate_default=True,
    )
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
    result_type: Union[type[T], GenericAlias, _LiteralGenericAlias, None] = None
    error: Union[str, None] = None
    tools: list[ToolType] = []
    user_access: bool = False
    is_auto_completed_by_subtasks: bool = False
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    _parent: "Union[Task, None]" = None
    _downstreams: list["Task"] = []
    model_config = dict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(
        self,
        objective=None,
        result_type=None,
        parent: "Task" = None,
        **kwargs,
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
        include_fields = [
            "id",
            "objective",
            "status",
            "result_type",
            "agents",
            "context",
            "user_access",
            "subtasks",
            "depends_on",
            "tools",
        ]
        fields = self.model_dump(include=include_fields)
        field_str = ", ".join(
            f"{k}={f'"{fields[k]}"' if isinstance(fields[k], str) else fields[k] }"
            for k in include_fields
        )
        return f"{self.__class__.__name__}({field_str})"

    @field_validator("agents", mode="before")
    def _default_agents(cls, v):
        from controlflow.core.agent import default_agent
        from controlflow.core.flow import get_flow

        if v is None:
            try:
                flow = get_flow()
            except ValueError:
                flow = None
            if flow and flow.agents:
                v = flow.agents
            else:
                v = [default_agent()]
        if not v:
            raise ValueError("At least one agent is required.")
        return v

    @field_validator("result_type", mode="before")
    def _turn_list_into_literal_result_type(cls, v):
        if isinstance(v, (list, tuple, set)):
            return Literal[tuple(v)]  # type: ignore
        return v

    @model_validator(mode="after")
    def _finalize(self):
        # validate correlated settings
        if self.result_type is not None and self.is_auto_completed_by_subtasks:
            raise ValueError(
                "Tasks with a result type cannot be auto-completed by their subtasks."
            )

        # create dependencies to tasks passed in as context
        context_tasks = collect_tasks(self.context)

        for task in context_tasks:
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
        if result_type is not None:
            return repr(result_type)

    @field_serializer("agents")
    def _serialize_agents(agents: list["Agent"]):
        return [
            a.model_dump(include={"name", "description", "tools", "user_access"})
            for a in agents
        ]

    @field_serializer("tools")
    def _serialize_tools(tools: list[ToolType]):
        return [
            marvin.utilities.tools.tool_from_function(t)
            if not isinstance(t, AssistantTool)
            else t
            for t in tools
        ]

    def friendly_name(self):
        if len(self.objective) > 50:
            objective = f'"{self.objective[:50]}..."'
        else:
            objective = f'"{self.objective}"'
        return f"Task {self.id} ({objective})"

    def as_graph(self) -> "Graph":
        from controlflow.core.graph import Graph

        return Graph.from_tasks(tasks=[self])

    def add_subtask(self, task: "Task"):
        """
        Indicate that this task has a subtask (which becomes an implicit dependency).
        """
        if task._parent is None:
            task._parent = self
        elif task._parent is not self:
            raise ValueError(f"{self.friendly_name()} already has a parent.")
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

    def run_once(self, agent: "Agent" = None):
        """
        Runs the task with provided agent. If no agent is provided, one will be selected from the task's agents.
        """
        from controlflow.core.controller import Controller

        controller = Controller(tasks=[self], agents=agent)

        controller.run_once()

    def run(self, raise_on_error: bool = True, max_iterations: int = NOTSET) -> T:
        """
        Runs the task with provided agents until it is complete.

        If max_iterations is provided, the task will run at most that many times before raising an error.
        """
        if max_iterations == NOTSET:
            max_iterations = controlflow.settings.max_task_iterations
        if max_iterations is None:
            max_iterations = float("inf")

        counter = 0
        while self.is_incomplete():
            if counter >= max_iterations:
                raise ValueError(
                    f"{self.friendly_name()} did not complete after {max_iterations} iterations."
                )
            self.run_once()
            counter += 1
        if self.is_successful():
            return self.result
        elif self.is_failed() and raise_on_error:
            raise ValueError(f"{self.friendly_name()} failed: {self.error}")

    @contextmanager
    def _context(self):
        stack = ctx.get("tasks", [])
        with ctx(tasks=stack + [self]):
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

    def get_tools(self) -> list[ToolType]:
        tools = self.tools.copy()
        if self.is_incomplete():
            tools.extend([self._create_fail_tool(), self._create_success_tool()])
            # add skip tool if this task has a parent task
            if self._parent is not None:
                tools.append(self._create_skip_tool())
        if self.user_access:
            tools.append(marvin.utilities.tools.tool_from_function(talk_to_human))
        return [wrap_prefect_tool(t) for t in tools]

    def dependencies(self):
        return self.depends_on + self.subtasks

    def mark_successful(self, result: T = None, validate: bool = True):
        if validate:
            if any(t.is_incomplete() for t in self.depends_on):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "upstream dependencies are completed. Incomplete dependencies "
                    f"are: {', '.join(t.friendly_name() for t in self.depends_on if t.is_incomplete())}"
                )
            elif any(t.is_incomplete() for t in self.subtasks):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "subtasks are completed. Incomplete subtasks "
                    f"are: {', '.join(t.friendly_name() for t in self.subtasks if t.is_incomplete())}"
                )

        if self.result_type is None and result is not None:
            raise ValueError(
                f"Task {self.objective} specifies no result type, but a result was provided."
            )
        elif self.result_type is not None:
            result = TypeAdapter(self.result_type).validate_python(result)

        self.result = result
        self.status = TaskStatus.SUCCESSFUL

        # attempt to complete the parent, if appropriate
        if (
            self._parent
            and self._parent.is_auto_completed_by_subtasks
            and all_complete(self._parent.dependencies())
        ):
            self._parent.mark_successful(validate=True)

        return f"{self.friendly_name()} marked successful. Updated task definition: {self.model_dump()}"

    def mark_failed(self, message: Union[str, None] = None):
        self.error = message
        self.status = TaskStatus.FAILED

        # attempt to fail the parent, if appropriate
        if (
            self._parent
            and self._parent.is_auto_completed_by_subtasks
            and all_complete(self._parent.dependencies())
        ):
            self._parent.mark_failed()

        return f"{self.friendly_name()} marked failed. Updated task definition: {self.model_dump()}"

    def mark_skipped(self):
        self.status = TaskStatus.SKIPPED
        # attempt to complete the parent, if appropriate
        if (
            self._parent
            and self._parent.is_auto_completed_by_subtasks
            and all_complete(self._parent.dependencies())
        ):
            self._parent.mark_successful(validate=False)

        return f"{self.friendly_name()} marked skipped. Updated task definition: {self.model_dump()}"


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


def task(
    fn=None,
    *,
    objective: str = None,
    instructions: str = None,
    agents: list["Agent"] = None,
    tools: list[ToolType] = None,
    user_access: bool = None,
):
    """
    A decorator that turns a Python function into a Task. The Task objective is
    set to the function name, and the instructions are set to the function
    docstring. When the function is called, the arguments are provided to the
    task as context, and the task is run to completion. If successful, the task
    result is returned; if failed, an error is raised.
    """

    if fn is None:
        return functools.partial(
            task,
            objective=objective,
            instructions=instructions,
            agents=agents,
            tools=tools,
            user_access=user_access,
        )

    sig = inspect.signature(fn)

    if objective is None:
        objective = fn.__name__

    if instructions is None:
        instructions = fn.__doc__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        task = Task(
            objective=objective,
            instructions=instructions,
            agents=agents,
            context=bound.arguments,
            result_type=fn.__annotations__.get("return"),
            user_access=user_access or False,
            tools=tools or [],
        )

        task.run()
        return task.result

    return wrapper
