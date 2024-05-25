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
    Optional,
    TypeVar,
    Union,
    _LiteralGenericAlias,
)

from pydantic import (
    Field,
    PydanticSchemaGenerationError,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

import controlflow
from controlflow.instructions import get_instructions
from controlflow.llm.tools import annotate_fn
from controlflow.tools.talk_to_human import talk_to_human
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.tasks import (
    collect_tasks,
    visit_task_collection,
)
from controlflow.utilities.types import (
    NOTSET,
    ControlFlowModel,
    PandasDataFrame,
    PandasSeries,
    ToolType,
)

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.flow import Flow
    from controlflow.core.graph import Graph

T = TypeVar("T")
logger = get_logger(__name__)


class TaskStatus(Enum):
    INCOMPLETE = "INCOMPLETE"
    SUCCESSFUL = "SUCCESSFUL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class Task(ControlFlowModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:5]))
    objective: str = Field(
        ..., description="A brief description of the required result."
    )
    instructions: Union[str, None] = Field(
        None, description="Detailed instructions for completing the task."
    )
    agents: Optional[list["Agent"]] = Field(
        None,
        description="The agents assigned to the task. If not provided, agents "
        "will be inferred from the parent task, flow, or global default.",
        validate_default=True,
    )
    context: dict = Field(
        default_factory=dict,
        description="Additional context for the task. If tasks are provided as "
        "context, they are automatically added as `depends_on`",
    )
    parent: Optional["Task"] = Field(
        None,
        description="The parent task of this task. Subtasks are considered"
        " upstream dependencies of their parents.",
        validate_default=True,
    )
    depends_on: set["Task"] = Field(
        default_factory=set, description="Tasks that this task depends on explicitly."
    )
    status: TaskStatus = TaskStatus.INCOMPLETE
    result: T = None
    result_type: Union[type[T], GenericAlias, _LiteralGenericAlias, None] = Field(
        None,
        description="The expected type of the result. This should be a type"
        ", generic alias, BaseModel subclass, pd.DataFrame, or pd.Series.",
    )
    error: Union[str, None] = None
    tools: list[Callable] = Field(
        default_factory=list,
        description="Tools available to every agent working on this task.",
    )
    user_access: bool = False
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    _subtasks: set["Task"] = set()
    _downstreams: set["Task"] = set()
    model_config = dict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(
        self,
        objective=None,
        result_type=None,
        **kwargs,
    ):
        # allow certain args to be provided as a positional args
        if result_type is not None:
            kwargs["result_type"] = result_type
        if objective is not None:
            kwargs["objective"] = objective

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions")
                or "" + "\n" + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)
        self.__cm_stack = []

    def __repr__(self):
        include_fields = [
            "id",
            "objective",
            "status",
            "is_ready",
            "result_type",
            "agents",
            "context",
            "user_access",
            "parent",
            "depends_on",
            "tools",
        ]
        fields = self.model_dump(include=include_fields)
        field_str = ", ".join(
            f'{k}="{fields[k]}"' if isinstance(fields[k], str) else f"{k}={fields[k]}"
            for k in include_fields
        )
        return f"{self.__class__.__name__}({field_str})"

    @field_validator("parent", mode="before")
    def _default_parent(cls, v):
        if v is None:
            parent_tasks = ctx.get("tasks", [])
            v = parent_tasks[-1] if parent_tasks else None

        return v

    @field_validator("agents", mode="before")
    def _validate_agents(cls, v):
        if v == []:
            raise ValueError("At least one agent is required.")
        return v

    @field_validator("result_type", mode="before")
    def _turn_list_into_literal_result_type(cls, v):
        if isinstance(v, (list, tuple, set)):
            return Literal[tuple(v)]  # type: ignore
        return v

    @model_validator(mode="after")
    def _finalize(self):
        from controlflow.core.flow import get_flow

        # add task to flow, if exists
        if flow := get_flow():
            flow.add_task(self)

        # create dependencies to tasks passed in as depends_on
        for task in self.depends_on:
            self.add_dependency(task)

        # create dependencies to tasks passed as subtasks
        if self.parent is not None:
            self.parent.add_subtask(self)

        # create dependencies to tasks passed in as context
        context_tasks = collect_tasks(self.context)

        for task in context_tasks:
            self.depends_on.add(task)

        return self

    @field_serializer("parent")
    def _serialize_parent(self, parent: Optional["Task"]):
        return parent.id if parent is not None else None

    @field_serializer("depends_on")
    def _serialize_depends_on(self, depends_on: set["Task"]):
        return [t.id for t in depends_on]

    @field_serializer("context")
    def _serialize_context(self, context: dict):
        def visitor(task):
            return f"<Result from task {task.id}>"

        return visit_task_collection(context, visitor)

    @field_serializer("result_type")
    def _serialize_result_type(self, result_type: list["Task"]):
        if result_type is not None:
            return repr(result_type)

    @field_serializer("agents")
    def _serialize_agents(self, agents: Optional[list["Agent"]]):
        agents = self.get_agents()
        return [
            a.model_dump(include={"name", "description", "tools", "user_access"})
            for a in agents
        ]

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[ToolType]):
        tools = controlflow.llm.tools.as_tools(tools)
        return [t.model_dump({"name", "description"}) for t in tools]

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
        if task.parent is None:
            task.parent = self
        elif task.parent is not self:
            raise ValueError(f"{self.friendly_name()} already has a parent.")
        self._subtasks.add(task)
        self.depends_on.add(task)

    def add_dependency(self, task: "Task"):
        """
        Indicate that this task depends on another task.
        """
        self.depends_on.add(task)
        task._downstreams.add(self)

    def run_once(self, agent: "Agent" = None, flow: "Flow" = None):
        """
        Runs the task with provided agent. If no agent is provided, one will be selected from the task's agents.
        """
        from controlflow.core.controller import Controller
        from controlflow.core.flow import get_flow

        # run once doesn't create new flows because the history would be lost
        flow = flow or get_flow()
        if flow is None:
            raise ValueError(
                "Task.run_once() must be called within a flow context or with a flow argument."
            )

        controller = Controller(tasks=[self], agents=agent, flow=flow)

        controller.run_once()

    def run(
        self,
        raise_on_error: bool = True,
        max_iterations: int = NOTSET,
        flow: "Flow" = None,
    ) -> T:
        """
        Runs the task with provided agents until it is complete.

        If max_iterations is provided, the task will run at most that many times before raising an error.
        """
        from controlflow.core.flow import Flow, get_flow

        if max_iterations == NOTSET:
            max_iterations = controlflow.settings.max_task_iterations
        if max_iterations is None:
            max_iterations = float("inf")

        flow = flow or get_flow()
        if flow is None:
            if controlflow.settings.strict_flow_context:
                raise ValueError(
                    "Task.run() must be called within a flow context or with a "
                    "flow argument if implicit flows are disabled."
                )
            else:
                flow = Flow()

        counter = 0
        while self.is_incomplete():
            if counter >= max_iterations:
                raise ValueError(
                    f"{self.friendly_name()} did not complete after {max_iterations} iterations."
                )
            self.run_once(flow=flow)
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
        # use stack so we can enter the context multiple times
        self.__cm_stack.append(self._context())
        return self.__cm_stack[-1].__enter__()

    def __exit__(self, *exc_info):
        return self.__cm_stack.pop().__exit__(*exc_info)

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

    @computed_field
    @property
    def is_ready(self) -> bool:
        """
        Returns True if all dependencies are complete and this task is incomplete.
        """
        return self.is_incomplete() and all(t.is_complete() for t in self.depends_on)

    def __hash__(self):
        return id(self)

    def _create_success_tool(self) -> Callable:
        """
        Create an agent-compatible tool for marking this task as successful.
        """
        # generate tool for result_type=None
        if self.result_type is None:

            def succeed() -> str:
                return self.mark_successful(result=None)

        # generate tool for other result types
        else:
            result_schema = generate_result_schema(self.result_type)

            def succeed(result: result_schema) -> str:  # type: ignore
                return self.mark_successful(result=result)

        return annotate_fn(
            succeed,
            name=f"mark_task_{self.id}_successful",
            description=f"Mark task {self.id} as successful.",
            metadata=dict(is_task_status_tool=True),
        )

    def _create_fail_tool(self) -> Callable:
        """
        Create an agent-compatible tool for failing this task.
        """

        return annotate_fn(
            self.mark_failed,
            name=f"mark_task_{self.id}_failed",
            description=f"Mark task {self.id} as failed. Only use when a technical issue like a broken tool or unresponsive human prevents completion.",
            metadata=dict(is_task_status_tool=True),
        )

    def _create_skip_tool(self) -> Callable:
        """
        Create an agent-compatible tool for skipping this task.
        """
        return annotate_fn(
            self.mark_skipped,
            name=f"mark_task_{self.id}_skipped",
            description=f"Mark task {self.id} as skipped. Only use when completing its parent task early.",
            metadata=dict(is_task_status_tool=True),
        )

    def get_agents(self) -> list["Agent"]:
        if self.agents:
            return self.agents
        elif self.parent:
            return self.parent.get_agents()
        else:
            from controlflow.core.agent import get_default_agent
            from controlflow.core.flow import get_flow

            try:
                flow = get_flow()
            except ValueError:
                flow = None
            if flow and flow.agents:
                return flow.agents
            else:
                return [get_default_agent()]

    def get_tools(self) -> list[Callable]:
        tools = self.tools.copy()
        if self.is_incomplete():
            tools.extend([self._create_fail_tool(), self._create_success_tool()])
            # add skip tool if this task has a parent task
            if self.parent is not None:
                tools.append(self._create_skip_tool())
        if self.user_access:
            tools.append(talk_to_human)
        return tools
        # return [wrap_prefect_tool(t) for t in tools]

    def set_status(self, status: TaskStatus):
        self.status = status
        if tui := ctx.get("tui"):
            tui.update_task(self)

    def mark_successful(self, result: T = None, validate_upstreams: bool = True):
        if validate_upstreams:
            if any(t.is_incomplete() for t in self.depends_on):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "upstream dependencies are completed. Incomplete dependencies "
                    f"are: {', '.join(t.friendly_name() for t in self.depends_on if t.is_incomplete())}"
                )
            elif any(t.is_incomplete() for t in self._subtasks):
                raise ValueError(
                    f"Task {self.objective} cannot be marked successful until all of its "
                    "subtasks are completed. Incomplete subtasks "
                    f"are: {', '.join(t.friendly_name() for t in self._subtasks if t.is_incomplete())}"
                )

        self.result = validate_result(result, self.result_type)
        self.set_status(TaskStatus.SUCCESSFUL)

        return f"{self.friendly_name()} marked successful."

    def mark_failed(self, message: Union[str, None] = None):
        self.error = message
        self.set_status(TaskStatus.FAILED)
        return f"{self.friendly_name()} marked failed."

    def mark_skipped(self):
        self.set_status(TaskStatus.SKIPPED)
        return f"{self.friendly_name()} marked skipped."


def generate_result_schema(result_type: type[T]) -> type[T]:
    result_schema = None
    # try loading pydantic-compatible schemas
    try:
        TypeAdapter(result_type)
        result_schema = result_type
    except PydanticSchemaGenerationError:
        pass
    # try loading as dataframe
    try:
        import pandas as pd

        if result_type is pd.DataFrame:
            result_schema = PandasDataFrame
        elif result_type is pd.Series:
            result_schema = PandasSeries
    except ImportError:
        pass
    if result_schema is None:
        raise ValueError(
            f"Could not load or infer schema for result type {result_type}. "
            "Please use a custom type or add compatibility."
        )
    return result_schema


def validate_result(result: Any, result_type: type[T]) -> T:
    if result_type is None and result is not None:
        raise ValueError("Task has result_type=None, but a result was provided.")
    elif result_type is not None:
        try:
            result = TypeAdapter(result_type).validate_python(result)
        except PydanticSchemaGenerationError:
            if isinstance(result, dict):
                result = result_type(**result)
            else:
                result = result_type(result)

        # Convert DataFrame schema back into pd.DataFrame object
        if result_type == PandasDataFrame:
            import pandas as pd

            result = pd.DataFrame(**result)
        elif result_type == PandasSeries:
            import pandas as pd

            result = pd.Series(**result)

    return result
