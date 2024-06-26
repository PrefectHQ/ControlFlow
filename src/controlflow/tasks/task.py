import datetime
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    GenericAlias,
    Literal,
    Optional,
    TypeVar,
    Union,
    _LiteralGenericAlias,
)

from prefect.context import TaskRunContext
from pydantic import (
    Field,
    PydanticSchemaGenerationError,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

import controlflow
from controlflow.agents import Agent
from controlflow.instructions import get_instructions
from controlflow.tools import Tool
from controlflow.tools.talk_to_user import talk_to_user
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import PrefectTrackingTask
from controlflow.utilities.prefect import prefect_task as prefect_task
from controlflow.utilities.tasks import (
    collect_tasks,
    visit_task_collection,
)
from controlflow.utilities.types import (
    NOTSET,
    ControlFlowModel,
)

if TYPE_CHECKING:
    from controlflow.controllers.graph import Graph
    from controlflow.flows import Flow

T = TypeVar("T")
logger = get_logger(__name__)


def get_task_run_name() -> str:
    context = TaskRunContext.get()
    return f'Run {context.parameters["self"].friendly_name()}'


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
        str,
        description="The expected type of the result. This should be a type"
        ", generic alias, BaseModel subclass, pd.DataFrame, or pd.Series. "
        "Can be None if no result is expected or the agent should communicate internally.",
    )
    error: Union[str, None] = None
    tools: list[Callable] = Field(
        default_factory=list,
        description="Tools available to every agent working on this task.",
    )
    user_access: bool = False
    private: bool = False
    agent_strategy: Optional[Callable] = Field(
        None,
        description="A function that returns an agent, used for customizing how "
        "the next agent is selected. The returned agent must be one "
        "of the assigned agents. If not provided, will be inferred "
        "from the parent task; round-robin selection is the default. "
        "Only used for tasks with more than one agent assigned.",
    )
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    _subtasks: set["Task"] = set()
    _downstreams: set["Task"] = set()
    _iteration: int = 0
    _cm_stack: list[contextmanager] = []
    _prefect_task: Optional[PrefectTrackingTask] = None

    model_config = dict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(
        self,
        objective: str = None,
        result_type: Any = NOTSET,
        infer_parent: bool = True,
        **kwargs,
    ):
        # allow certain args to be provided as a positional args
        if result_type is not NOTSET:
            kwargs["result_type"] = result_type
        if objective is not None:
            kwargs["objective"] = objective
        # if parent is None and infer parent is False, set parent to NOTSET
        if not infer_parent and kwargs.get("parent") is None:
            kwargs["parent"] = NOTSET
        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions")
                or "" + "\n" + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)

        self._prefect_task = PrefectTrackingTask(
            name=f"Working on {self.friendly_name()}...",
            description=self.instructions,
            tags=[self.__class__.__name__],
        )

    def __hash__(self):
        return hash((self.__class__.__name__, self.id))

    def __eq__(self, other):
        """
        Tasks have set attributes and set equality is based on id() of their
        contents, not equality of objects. This means that two tasks are not
        equal unless their set attributes satisfy an identity criteria, which is
        too strict.
        """
        if type(self) == type(other):
            d1 = dict(self)
            d2 = dict(other)
            # conver sets to lists for comparison
            d1["depends_on"] = list(d1["depends_on"])
            d2["depends_on"] = list(d2["depends_on"])
            return d1 == d2
        return False

    def __repr__(self) -> str:
        serialized = self.model_dump()
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in serialized.items())})"

    @field_validator("parent", mode="before")
    def _default_parent(cls, v):
        if v is None:
            parent_tasks = ctx.get("tasks", [])
            v = parent_tasks[-1] if parent_tasks else None
        elif v is NOTSET:
            v = None
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
        # add task to flow, if exists
        if flow := controlflow.flows.get_flow():
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
        if result_type is None:
            return None
        try:
            schema = TypeAdapter(result_type).json_schema()
        except PydanticSchemaGenerationError:
            schema = "<schema could not be generated>"

        return dict(type=repr(result_type), schema=schema)

    @field_serializer("agents")
    def _serialize_agents(self, agents: Optional[list["Agent"]]):
        agents = self.get_agents()
        return [a.serialize_for_prompt() for a in agents]

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        tools = controlflow.tools.as_tools(tools)
        # tools are Pydantic 1 objects
        return [t.dict(include={"name", "description"}) for t in tools]

    def friendly_name(self):
        if len(self.objective) > 50:
            objective = f'"{self.objective[:50]}..."'
        else:
            objective = f'"{self.objective}"'
        return f"Task {self.id} ({objective})"

    def as_graph(self) -> "Graph":
        from controlflow.controllers.graph import Graph

        return Graph.from_tasks(tasks=[self])

    @property
    def subtasks(self) -> list["Task"]:
        from controlflow.controllers.graph import Graph

        return Graph.from_tasks(tasks=self._subtasks).topological_sort()

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

    def run_once(self, agents: Optional[list["Agent"]] = None, flow: "Flow" = None):
        """
        Runs the task with provided agent. If no agent is provided, one will be selected from the task's agents.
        """
        # run once doesn't create new flows because the history would be lost
        flow = flow or controlflow.flows.get_flow()
        if flow is None:
            raise ValueError(
                "Task.run_once() must be called within a flow context or with a flow argument."
            )

        from controlflow.controllers import Controller

        controller = Controller(
            tasks=[self], flow=flow, agents={self: agents} if agents else None
        )
        controller.run_once()

    async def run_once_async(
        self, agents: Optional[list["Agent"]] = None, flow: "Flow" = None
    ):
        """
        Runs the task with provided agent. If no agent is provided, one will be selected from the task's agents.
        """

        # run once doesn't create new flows because the history would be lost
        flow = flow or controlflow.flows.get_flow()
        if flow is None:
            raise ValueError(
                "Task.run_once_async() must be called within a flow context or with a flow argument."
            )

        from controlflow.controllers import Controller

        controller = Controller(
            tasks=[self], flow=flow, agents={self: agents} if agents else None
        )
        await controller.run_once_async()

    @prefect_task(task_run_name=get_task_run_name)
    def run(
        self,
        agents: Optional[list["Agent"]] = None,
        raise_on_error: bool = True,
        flow: "Flow" = None,
    ) -> Generator[T, None, None]:
        """
        Internal function that can handle both sync and async runs by yielding either the result or the coroutine.
        """
        from controlflow.flows import Flow, get_flow

        flow = flow or get_flow()
        if flow is None:
            if controlflow.settings.strict_flow_context:
                raise ValueError(
                    "Task.run() must be called within a flow context or with a "
                    "flow argument if implicit flows are disabled."
                )
            else:
                flow = Flow()

        from controlflow.controllers import Controller

        controller = Controller(
            tasks=[self], flow=flow, agents={self: agents} if agents else None
        )
        controller.run()

        if self.is_successful():
            return self.result
        elif self.is_failed() and raise_on_error:
            raise ValueError(f"{self.friendly_name()} failed: {self.error}")

    @prefect_task(task_run_name=get_task_run_name)
    async def run_async(
        self,
        agents: Optional[list["Agent"]] = None,
        raise_on_error: bool = True,
        flow: "Flow" = None,
    ) -> Generator[T, None, None]:
        """
        Internal function that can handle both sync and async runs by yielding either the result or the coroutine.
        """
        from controlflow.flows import Flow, get_flow

        flow = flow or get_flow()
        if flow is None:
            if controlflow.settings.strict_flow_context:
                raise ValueError(
                    "Task.run() must be called within a flow context or with a "
                    "flow argument if implicit flows are disabled."
                )
            else:
                flow = Flow()

        from controlflow.controllers import Controller

        controller = Controller(
            tasks=[self], flow=flow, agents={self: agents} if agents else None
        )
        await controller.run_async()

        if self.is_successful():
            return self.result
        elif self.is_failed() and raise_on_error:
            raise ValueError(f"{self.friendly_name()} failed: {self.error}")

    def copy(self):
        return self.model_copy()

    @contextmanager
    def create_context(self):
        stack = ctx.get("tasks", [])
        with ctx(tasks=stack + [self]):
            yield self

    def __enter__(self):
        # use stack so we can enter the context multiple times
        self._cm_stack.append(self.create_context())
        return self._cm_stack[-1].__enter__()

    def __exit__(self, *exc_info):
        return self._cm_stack.pop().__exit__(*exc_info)

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

    def _create_success_tool(self) -> Tool:
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

        return Tool.from_function(
            succeed,
            name=f"mark_task_{self.id}_successful",
            description=f"Mark task {self.id} as successful.",
            metadata=dict(ignore_result=True),
        )

    def _create_fail_tool(self) -> Tool:
        """
        Create an agent-compatible tool for failing this task.
        """

        return Tool.from_function(
            self.mark_failed,
            name=f"mark_task_{self.id}_failed",
            description=f"Mark task {self.id} as failed. Only use when technical errors prevent success.",
            metadata=dict(ignore_result=True),
        )

    def _create_skip_tool(self) -> Tool:
        """
        Create an agent-compatible tool for skipping this task.
        """
        return Tool.from_function(
            self.mark_skipped,
            name=f"mark_task_{self.id}_skipped",
            description=f"Mark task {self.id} as skipped. Only use when completing a parent task early.",
            metadata=dict(ignore_result=True),
        )

    def get_agents(self) -> list["Agent"]:
        if self.agents:
            return self.agents
        elif self.parent:
            return self.parent.get_agents()
        else:
            from controlflow.agents import get_default_agent
            from controlflow.flows import get_flow

            try:
                flow = get_flow()
            except ValueError:
                flow = None
            if flow and flow.agents:
                return flow.agents
            else:
                return [get_default_agent()]

    def get_agent_strategy(self) -> Callable:
        """
        Get a function for selecting the next agent to work on this
        task.

        If an agent_strategy is provided, it will be used. Otherwise, the parent
        task's agent_strategy will be used. Finally, the global default agent_strategy
        will be used (round-robin selection).
        """
        if self.agent_strategy is not None:
            return self.agent_strategy
        elif self.parent:
            return self.parent.get_agent_strategy()
        else:
            import controlflow.tasks.agent_strategies

            return controlflow.tasks.agent_strategies.round_robin

    def get_tools(self) -> list[Union[Tool, Callable]]:
        tools = self.tools.copy()
        # if this task is ready to run, generate tools
        if self.is_ready():
            tools.extend([self._create_fail_tool(), self._create_success_tool()])
            # add skip tool if this task has a parent task
            # if self.parent is not None:
            #     tools.append(self._create_skip_tool())
        if self.user_access:
            tools.append(talk_to_user)
        return tools
        # return [wrap_prefect_tool(t) for t in tools]

    def set_status(self, status: TaskStatus):
        self.status = status

        # update TUI
        if tui := ctx.get("tui"):
            tui.update_task(self)

        # update Prefect
        if self._prefect_task.is_started:
            if status == TaskStatus.SUCCESSFUL:
                self._prefect_task.succeed(self.result)
            elif status == TaskStatus.FAILED:
                self._prefect_task.fail(self.error)
            elif status == TaskStatus.SKIPPED:
                self._prefect_task.skip()

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

        if agent := ctx.get("agent"):
            return f"{self.friendly_name()} marked successful by {agent.name}."
        return f"{self.friendly_name()} marked successful."

    def mark_failed(self, message: Union[str, None] = None):
        self.error = message
        self.set_status(TaskStatus.FAILED)
        if agent := ctx.get("agent"):
            return f"{self.friendly_name()} marked failed by {agent.name}."
        return f"{self.friendly_name()} marked failed."

    def mark_skipped(self):
        self.set_status(TaskStatus.SKIPPED)
        if agent := ctx.get("agent"):
            return f"{self.friendly_name()} marked skipped by {agent.name}."
        return f"{self.friendly_name()} marked skipped."

    def generate_subtasks(self, instructions: str = None, agent: Agent = None):
        """
        Generate subtasks for this task based on the provided instructions.
        Subtasks can reuse the same tools and agents as this task.
        """
        from controlflow.planning.plan import create_plan

        # enter a context to set the parent task
        with self:
            create_plan(
                self.objective,
                instructions=instructions,
                planning_agent=agent or self.agents,
                agents=self.agents,
                tools=self.tools,
                context=self.context,
            )


def generate_result_schema(result_type: type[T]) -> type[T]:
    result_schema = None
    # try loading pydantic-compatible schemas
    try:
        TypeAdapter(result_type)
        result_schema = result_type
    except PydanticSchemaGenerationError:
        pass
    # try loading as dataframe
    # try:
    #     import pandas as pd

    #     if result_type is pd.DataFrame:
    #         result_schema = PandasDataFrame
    #     elif result_type is pd.Series:
    #         result_schema = PandasSeries
    # except ImportError:
    #     pass
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
        # if result_type == PandasDataFrame:
        #     import pandas as pd

        #     result = pd.DataFrame(**result)
        # elif result_type == PandasSeries:
        #     import pandas as pd

        #     result = pd.Series(**result)

    return result
