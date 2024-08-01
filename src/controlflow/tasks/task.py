import datetime
from contextlib import ExitStack, contextmanager
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

from prefect.context import TaskRunContext
from pydantic import (
    Field,
    PydanticSchemaGenerationError,
    TypeAdapter,
    field_serializer,
    field_validator,
)

import controlflow
from controlflow.agents import BaseAgent
from controlflow.instructions import get_instructions
from controlflow.tools import Tool
from controlflow.tools.talk_to_user import talk_to_user
from controlflow.utilities.context import ctx
from controlflow.utilities.general import (
    NOTSET,
    ControlFlowModel,
    hash_objects,
)
from controlflow.utilities.logging import deprecated, get_logger
from controlflow.utilities.prefect import PrefectTrackingTask
from controlflow.utilities.prefect import prefect_task as prefect_task
from controlflow.utilities.tasks import (
    collect_tasks,
    visit_task_collection,
)

if TYPE_CHECKING:
    from controlflow.flows import Flow
    from controlflow.orchestration.agent_context import AgentContext

T = TypeVar("T")
logger = get_logger(__name__)


def get_task_run_name() -> str:
    context = TaskRunContext.get()
    return f'Run {context.parameters["self"].friendly_name()}'


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESSFUL = "SUCCESSFUL"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


INCOMPLETE_STATUSES = {TaskStatus.PENDING, TaskStatus.RUNNING}
COMPLETE_STATUSES = {TaskStatus.SUCCESSFUL, TaskStatus.FAILED, TaskStatus.SKIPPED}


class Task(ControlFlowModel):
    id: str = None
    objective: str = Field(
        ..., description="A brief description of the required result."
    )
    instructions: Union[str, None] = Field(
        None, description="Detailed instructions for completing the task."
    )
    agent: Optional[BaseAgent] = Field(
        None,
        description="The agent or team of agents assigned to the task. "
        "If not provided, it will be inferred from the parent task, flow, or global default.",
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
    prompt: Optional[str] = Field(
        None,
        description="A prompt to display to the agent working on the task. "
        "Prompts are formatted as jinja templates, with keywords `task: Task` and `context: AgentContext`.",
    )
    status: TaskStatus = TaskStatus.PENDING
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
    private: bool = Field(
        False,
        description="Work on private tasks is not visible to agents other than those assigned to the task.",
    )
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    max_iterations: Optional[int] = Field(
        default_factory=lambda: controlflow.settings.max_task_iterations,
        description="The maximum number of iterations to attempt to run a task.",
    )
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
        # TODO: deprecated July 2024
        agent: Optional["BaseAgent"] = None,
        agents: Optional[list["BaseAgent"]] = None,
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

        if agent and agents:
            raise ValueError(
                "The 'agent' argument cannot be used with the 'agents' argument."
            )
        elif agents:
            kwargs["agent"] = agents
        else:
            kwargs["agent"] = agent

        super().__init__(**kwargs)

        self._prefect_task = PrefectTrackingTask(
            name=f"Working on {self.friendly_name()}...",
            description=self.instructions,
            tags=[self.__class__.__name__],
        )

        # create dependencies to tasks passed in as depends_on
        for task in self.depends_on:
            self.add_dependency(task)

        # create dependencies to tasks passed as subtasks
        if self.parent is not None:
            self.parent.add_subtask(self)

        # create dependencies to tasks passed in as context
        context_tasks = collect_tasks(self.context)

        for task in context_tasks:
            self.add_dependency(task)

        # add task to flow, if exists
        if flow := controlflow.flows.get_flow():
            flow.add_task(self)

        if self.id is None:
            self.id = self._generate_id()

    def _generate_id(self):
        return hash_objects(
            (
                type(self).__name__,
                self.objective,
                self.instructions,
                str(self.result_type),
                self.prompt,
                self.private,
                str(self.context),
            )
        )

    def __hash__(self) -> int:
        return id(self)

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
        serialized = self.model_dump(include={"id", "objective"})
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in serialized.items())})"

    @field_validator("agent", mode="before")
    def _validate_agent(cls, v):
        if isinstance(v, list):
            if len(v) > 1:
                v = controlflow.defaults.team(agents=v)
            elif v:
                v = v[0]
            else:
                v = None
        return v

    @field_validator("parent", mode="before")
    def _default_parent(cls, v):
        if v is None:
            parent_tasks = ctx.get("tasks", [])
            v = parent_tasks[-1] if parent_tasks else None
        elif v is NOTSET:
            v = None
        return v

    @field_validator("result_type", mode="before")
    def _turn_list_into_literal_result_type(cls, v):
        if isinstance(v, (list, tuple, set)):
            return Literal[tuple(v)]  # type: ignore
        return v

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

    @field_serializer("agent")
    def _serialize_agents(self, agent: Optional[BaseAgent]):
        return self.get_agent().serialize_for_prompt()

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        return [t.serialize_for_prompt() for t in controlflow.tools.as_tools(tools)]

    def friendly_name(self):
        if len(self.objective) > 50:
            objective = f'"{self.objective[:50]}..."'
        else:
            objective = f'"{self.objective}"'
        return f"Task {self.id} ({objective})"

    def serialize_for_prompt(self) -> dict:
        """
        Generate a prompt to share information about the task, for use in another object's prompt (like Flow)
        """
        return self.model_dump_json()

    @property
    def subtasks(self) -> list["Task"]:
        from controlflow.flows.graph import Graph

        return Graph(tasks=self._subtasks).topological_sort()

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

    @prefect_task(task_run_name=get_task_run_name)
    def run(
        self,
        steps: Optional[int] = None,
        agent: Optional[BaseAgent] = None,
        raise_on_error: bool = True,
        flow: "Flow" = None,
    ) -> T:
        """
        Run the task for the specified number of steps or until it is complete
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
                if steps:
                    logger.warning(
                        "Running a task with a steps argument but no flow is not "
                        "recommended, because the agent's history will be lost."
                    )
                flow = Flow()

        from controlflow.orchestration import Orchestrator

        orchestrator = Orchestrator(
            tasks=[self], flow=flow, agents={self: agent} if agent else None
        )
        orchestrator.run(steps=steps)

        if self.is_successful():
            return self.result
        elif self.is_failed() and raise_on_error:
            raise ValueError(f"{self.friendly_name()} failed: {self.error}")

    @prefect_task(task_run_name=get_task_run_name)
    async def run_async(
        self,
        steps: Optional[int] = None,
        agent: Optional[BaseAgent] = None,
        raise_on_error: bool = True,
        flow: "Flow" = None,
    ) -> T:
        """
        Run the task for the specified number of steps or until it is complete
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
                if steps:
                    logger.warning(
                        "Running a task with a steps argument but no flow is not "
                        "recommended, because the agent's history will be lost."
                    )
                flow = Flow()

        from controlflow.orchestration import Orchestrator

        orchestrator = Orchestrator(
            tasks=[self], flow=flow, agents={self: agent} if agent else None
        )
        await orchestrator.run_async(steps=steps)

        if self.is_successful():
            return self.result
        elif self.is_failed() and raise_on_error:
            raise ValueError(f"{self.friendly_name()} failed: {self.error}")

    @contextmanager
    def create_context(self):
        stack = ctx.get("tasks", [])
        with ctx(tasks=stack + [self]):
            yield self

    def __enter__(self):
        # use stack so we can enter the context multiple times
        self._cm_stack.append(ExitStack())
        return self._cm_stack[-1].enter_context(self.create_context())

    def __exit__(self, *exc_info):
        return self._cm_stack.pop().close()

    def is_incomplete(self) -> bool:
        return self.status in INCOMPLETE_STATUSES

    def is_complete(self) -> bool:
        return self.status in COMPLETE_STATUSES

    def is_pending(self) -> bool:
        return self.status == TaskStatus.PENDING

    def is_running(self) -> bool:
        return self.status == TaskStatus.RUNNING

    def is_successful(self) -> bool:
        return self.status == TaskStatus.SUCCESSFUL

    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def is_skipped(self) -> bool:
        return self.status == TaskStatus.SKIPPED

    def is_ready(self) -> bool:
        """
        Returns True if all dependencies are complete and this task is
        incomplete, meaning it is ready to be worked on.
        """
        return self.is_incomplete() and all(t.is_complete() for t in self.depends_on)

    def get_agent(self) -> BaseAgent:
        if self.agent:
            return self.agent
        elif self.parent:
            return self.parent.get_agent()
        else:
            from controlflow.flows import get_flow

            try:
                flow = get_flow()
            except ValueError:
                flow = None
            if flow and flow.agent:
                return flow.agent
            else:
                return controlflow.defaults.agent

    def get_tools(self) -> list[Union[Tool, Callable]]:
        tools = self.tools.copy()
        if self.user_access:
            tools.append(talk_to_user)
        return tools

    def get_prompt(self, context: "AgentContext") -> str:
        """
        Generate a prompt to share information about the task with an agent.
        """
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.TaskTemplate(
            template=self.prompt, task=self, context=context
        )
        return template.render()

    def set_status(self, status: TaskStatus):
        self.status = status

        # update TUI
        if tui := ctx.get("tui"):
            tui.update_task(self)

        # update Prefect
        if not self._prefect_task.is_started and status == TaskStatus.RUNNING:
            self._prefect_task.start(depends_on=[t.result for t in self.depends_on])
        elif self._prefect_task.is_started:
            if status == TaskStatus.SUCCESSFUL:
                self._prefect_task.succeed(self.result)
            elif status == TaskStatus.FAILED:
                self._prefect_task.fail(self.error)
            elif status == TaskStatus.SKIPPED:
                self._prefect_task.skip()

    def mark_running(self):
        self.set_status(TaskStatus.RUNNING)

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

    def mark_failed(self, reason: Optional[str] = None):
        self.error = reason
        self.set_status(TaskStatus.FAILED)

    def mark_skipped(self):
        self.set_status(TaskStatus.SKIPPED)

    def generate_subtasks(self, instructions: str = None, agent: BaseAgent = None):
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
                planning_agent=agent or self.agent,
                agents=[self.agent],
                tools=self.tools,
                context=self.context,
            )

    # Deprecated ---------------------------

    @deprecated("Use Task.run(steps=1) instead.", version="0.9")
    def run_once(self, *args, **kwargs):
        return self.run(*args, steps=1, **kwargs)

    @deprecated("Use Task.run_async(steps=1) instead.", version="0.9")
    async def run_once_async(self, *args, **kwargs):
        return await self.run_async(*args, steps=1, **kwargs)


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
