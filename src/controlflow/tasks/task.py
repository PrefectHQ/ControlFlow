import datetime
import textwrap
import warnings
from contextlib import ExitStack, contextmanager
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    GenericAlias,
    Iterator,
    Literal,
    Optional,
    TypeVar,
    Union,
    _AnnotatedAlias,
    _GenericAlias,
    _LiteralGenericAlias,
    _SpecialGenericAlias,
)
from uuid import uuid4

from prefect.context import TaskRunContext
from pydantic import (
    BaseModel,
    Field,
    PydanticSchemaGenerationError,
    RootModel,
    TypeAdapter,
    field_serializer,
    field_validator,
)
from pydantic_extra_types.pendulum_dt import DateTime
from typing_extensions import Self

import controlflow
from controlflow.agents import Agent
from controlflow.instructions import get_instructions
from controlflow.memory.async_memory import AsyncMemory
from controlflow.memory.memory import Memory
from controlflow.tools import Tool, tool
from controlflow.tools.input import cli_input
from controlflow.tools.tools import as_tools
from controlflow.utilities.context import ctx
from controlflow.utilities.general import (
    NOTSET,
    ControlFlowModel,
    hash_objects,
    safe_issubclass,
    unwrap,
)
from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.events.events import Event
    from controlflow.flows import Flow
    from controlflow.orchestration.handler import AsyncHandler, Handler
    from controlflow.orchestration.turn_strategies import TurnStrategy
    from controlflow.stream import Stream

T = TypeVar("T")
logger = get_logger(__name__)

COMPLETION_TOOLS = Literal["SUCCEED", "FAIL"]


class Labels(RootModel):
    root: tuple[Any, ...]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __repr__(self) -> str:
        return f"Labels: {', '.join(self.root)}"


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
    name: Optional[str] = Field(None, description="A name for the task.")
    objective: str = Field(
        ..., description="A brief description of the required result."
    )
    instructions: Union[str, None] = Field(
        None, description="Detailed instructions for completing the task."
    )
    agents: Optional[list[Agent]] = Field(
        default=None,
        description="A list of agents assigned to the task. "
        "If not provided, it will be inferred from the caller, parent task, flow, or global default.",
    )
    context: dict = Field(
        default_factory=dict,
        description="Additional context for the task.",
    )
    parent: Optional["Task"] = Field(
        NOTSET,
        description="The parent task of this task. Subtasks are considered"
        " upstream dependencies of their parents.",
        validate_default=True,
    )
    depends_on: set["Task"] = Field(
        default_factory=set, description="Tasks that this task depends on explicitly."
    )
    prompt: Optional[str] = Field(
        None, description="A prompt to display to the agent working on the task."
    )
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Union[T, str]] = None
    result_type: Union[
        type[T],
        GenericAlias,
        _GenericAlias,
        _SpecialGenericAlias,
        _AnnotatedAlias,
        Labels,
        None,
    ] = Field(
        NOTSET,
        description="The expected type of the result. This should be a type"
        ", generic alias, BaseModel subclass, or list of choices. "
        "Can be None if no result is expected or the agent should communicate internally.",
        validate_default=True,
    )
    result_validator: Optional[Callable] = Field(
        None,
        description="A function that validates the result. This should be a "
        "function that takes the raw result and either returns a validated "
        "result or raises an informative error if the result is not valid. The "
        "result validator function is called *after* the `result_type` is "
        "processed.",
    )
    tools: list[Tool] = Field(
        default_factory=list,
        description="Tools available to every agent working on this task.",
    )
    completion_tools: Optional[list[COMPLETION_TOOLS]] = Field(
        default=None,
        description="""
            Completion tools that will be generated for this task. If None, all 
            tools will be generated; if a list of strings, only the corresponding 
            tools will be generated automatically.
            """,
    )
    completion_agents: Optional[list[Agent]] = Field(
        default=None,
        description="Agents that are allowed to mark this task as complete. If None, all agents are allowed.",
    )
    interactive: bool = False
    memories: list[Union[Memory, AsyncMemory]] = Field(
        default=[],
        description="A list of memory modules for the task to use.",
    )
    max_llm_calls: Optional[int] = Field(
        default_factory=lambda: controlflow.settings.task_max_llm_calls,
        description="Maximum number of LLM calls to make before the task should be marked as failed. "
        "The total calls are measured over the life of the task, and include any LLM call for "
        "which this task is considered `assigned`.",
    )
    created_at: DateTime = Field(default_factory=datetime.datetime.now)
    wait_for_subtasks: bool = Field(
        default=True,
        description="If True, the task will not be considered ready until all subtasks are complete.",
    )
    _subtasks: set["Task"] = set()
    _downstreams: set["Task"] = set()
    _cm_stack: list[contextmanager] = []
    _llm_calls: int = 0

    model_config = dict(extra="forbid", arbitrary_types_allowed=True)

    def __init__(
        self,
        objective: str = None,
        user_access: bool = None,
        **kwargs,
    ):
        """
        Initialize a Task object.

        Args:
            objective (str, optional): The objective of the task. Defaults to None.
            result_type (Any, optional): The type of the result. Defaults to NOTSET.
            user_access (bool, optional): Whether the task is interactive. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # allow certain args to be provided as a positional args
        if objective is not None:
            kwargs["objective"] = objective

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions")
                or "" + "\n" + "\n".join(additional_instructions)
            ).strip()

        # deprecated in 0.9
        if user_access:
            warnings.warn(
                "The `user_access` argument is deprecated. Use `interactive=True` instead.",
                DeprecationWarning,
            )
            kwargs["interactive"] = True

        super().__init__(**kwargs)

        # create dependencies to tasks passed in as depends_on
        for task in self.depends_on:
            self.add_dependency(task)

        # create dependencies to tasks passed as subtasks
        if self.parent is not None:
            self.parent.add_subtask(self)

        if self.id is None:
            self.id = self._generate_id()

    def _generate_id(self):
        return str(uuid4())[:8]
        # generate a short, semi-stable ID for a task
        # return hash_objects(
        #     (
        #         type(self).__name__,
        #         self.objective,
        #         self.instructions,
        #         str(self.result_type),
        #         self.prompt,
        #         str(self.context),
        #     )
        # )

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other):
        """
        Tasks have set attributes and set equality is based on id() of their
        contents, not equality of objects. This means that two tasks are not
        equal unless their set attributes satisfy an identity criteria, which is
        too strict.
        """
        if type(self) is type(other):
            d1 = dict(self)
            d2 = dict(other)

            for attr in ["id", "created_at"]:
                d1.pop(attr)
                d2.pop(attr)

            # conver sets to lists for comparison
            d1["depends_on"] = list(d1["depends_on"])
            d2["depends_on"] = list(d2["depends_on"])
            d1["subtasks"] = list(self.subtasks)
            d2["subtasks"] = list(other.subtasks)
            return d1 == d2
        return False

    def __repr__(self) -> str:
        serialized = self.model_dump(include={"id", "objective"})
        return f"{self.__class__.__name__}({', '.join(f'{key}={repr(value)}' for key, value in serialized.items())})"

    @field_validator("objective")
    def _validate_objective(cls, v):
        if v:
            v = unwrap(v)
        return v

    @field_validator("instructions")
    def _validate_instructions(cls, v):
        if v:
            v = unwrap(v)
        return v

    @field_validator("agents")
    def _validate_agents(cls, v):
        if isinstance(v, list) and not v:
            raise ValueError("Agents must be `None` or a non-empty list of agents.")
        return v

    @field_validator("parent", mode="before")
    def _default_parent(cls, v):
        if v == NOTSET:
            parent_tasks = ctx.get("tasks", [])
            v = parent_tasks[-1] if parent_tasks else None
        return v

    @field_validator("result_type", mode="before")
    def _validate_result_type(cls, v):
        if v == NOTSET:
            v = str
        if isinstance(v, _LiteralGenericAlias):
            v = v.__args__
        if isinstance(v, (list, tuple, set)):
            v = Labels(v)
        return v

    @field_validator("tools", mode="before")
    def _validate_tools(cls, v):
        return as_tools(v or [])

    @field_serializer("parent")
    def _serialize_parent(self, parent: Optional["Task"]):
        return parent.id if parent is not None else None

    @field_serializer("depends_on")
    def _serialize_depends_on(self, depends_on: set["Task"]):
        return [t.id for t in depends_on]

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
    def _serialize_agents(self, agents: list[Agent]):
        return [agent.serialize_for_prompt() for agent in self.get_agents()]

    @field_serializer("completion_agents")
    def _serialize_completion_agents(self, completion_agents: Optional[list[Agent]]):
        if completion_agents is not None:
            return [agent.serialize_for_prompt() for agent in completion_agents]
        else:
            return None

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        return [t.serialize_for_prompt() for t in controlflow.tools.as_tools(tools)]

    def friendly_name(self):
        if self.name:
            name = self.name
        elif len(self.objective) > 50:
            name = f'"{self.objective[:50]}..."'
        else:
            name = f'"{self.objective}"'
        return f"Task #{self.id} ({name})"

    def serialize_for_prompt(self) -> dict:
        """
        Generate a prompt to share information about the task, for use in another object's prompt (like Flow)
        """
        return self.model_dump_json()

    @property
    def subtasks(self) -> list["Task"]:
        return list(sorted(self._subtasks, key=lambda t: t.created_at))

    # def subtask(self, **kwargs) -> "Task":
    #     task = Task(**kwargs)
    #     self.add_subtask(task)
    #     return task

    def add_subtask(self, task: "Task"):
        """
        Indicate that this task has a subtask (which becomes an implicit dependency).
        """
        if task.parent is None:
            task.parent = self
        elif task.parent is not self:
            raise ValueError(f"{self.friendly_name()} already has a parent.")
        self._subtasks.add(task)

    def add_dependency(self, task: "Task"):
        """
        Indicate that this task depends on another task.
        """
        self.depends_on.add(task)
        task._downstreams.add(self)

    def run(
        self,
        agent: Optional[Agent] = None,
        flow: "Flow" = None,
        turn_strategy: "TurnStrategy" = None,
        max_llm_calls: int = None,
        max_agent_turns: int = None,
        handlers: list["Handler"] = None,
        raise_on_failure: bool = True,
        model_kwargs: Optional[dict] = None,
        stream: Union[bool, "Stream"] = False,
    ) -> Union[T, Iterator[tuple["Event", Any, Optional[Any]]]]:
        """
        Run the task

        Args:
            agent: Optional agent to run the task
            flow: Optional flow to run the task in
            turn_strategy: Optional turn strategy to use
            max_llm_calls: Maximum number of LLM calls to make
            max_agent_turns: Maximum number of agent turns to make
            handlers: Optional list of handlers
            raise_on_failure: Whether to raise on task failure
            model_kwargs: Optional kwargs to pass to the model
            stream: If True, stream all events. Can also provide StreamFilter flags.

        Returns:
            If not streaming: The task result
            If streaming: Iterator of (event, snapshot, delta) tuples
        """
        result = controlflow.run_tasks(
            tasks=[self],
            flow=flow,
            agent=agent,
            turn_strategy=turn_strategy,
            max_llm_calls=max_llm_calls,
            max_agent_turns=max_agent_turns,
            raise_on_failure=False,
            handlers=handlers,
            model_kwargs=model_kwargs,
            stream=stream,
        )

        if stream:
            return result
        elif self.is_successful():
            return self.result
        elif raise_on_failure and self.is_failed():
            raise ValueError(f"{self.friendly_name()} failed: {self.result}")

    async def run_async(
        self,
        agent: Optional[Agent] = None,
        flow: "Flow" = None,
        turn_strategy: "TurnStrategy" = None,
        max_llm_calls: int = None,
        max_agent_turns: int = None,
        handlers: list[Union["Handler", "AsyncHandler"]] = None,
        raise_on_failure: bool = True,
        stream: Union[bool, "Stream"] = False,
    ) -> T:
        """
        Run the task
        """

        result = await controlflow.run_tasks_async(
            tasks=[self],
            flow=flow,
            agent=agent,
            turn_strategy=turn_strategy,
            max_llm_calls=max_llm_calls,
            max_agent_turns=max_agent_turns,
            raise_on_failure=False,
            handlers=handlers,
            stream=stream,
        )

        if stream:
            return result
        elif self.is_successful():
            return self.result
        elif raise_on_failure and self.is_failed():
            raise ValueError(f"{self.friendly_name()} failed: {self.result}")

    @contextmanager
    def create_context(self) -> Generator[Self, None, None]:
        stack = ctx.get("tasks") or []
        with ctx(tasks=stack + [self]):
            yield self

    def __enter__(self) -> Self:
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
        depends_on = self.depends_on
        if self.wait_for_subtasks:
            depends_on = depends_on.union(self._subtasks)

        return self.is_incomplete() and all(t.is_complete() for t in depends_on)

    def get_agents(self) -> list[Agent]:
        if self.agents is not None:
            return self.agents
        elif self.parent:
            return self.parent.get_agents()
        else:
            from controlflow.flows import get_flow

            try:
                flow = get_flow()
            except ValueError:
                flow = None
            if flow and flow.default_agent:
                return [flow.default_agent]
            else:
                return [controlflow.defaults.agent]

    def get_tools(self) -> list[Tool]:
        """
        Return a list of all tools available for the task.

        Note this does not include completion tools, which are handled separately.
        """
        tools = self.tools.copy()
        if self.interactive:
            tools.append(cli_input)
        for memory in self.memories:
            tools.extend(memory.get_tools())
        return as_tools(tools)

    def get_completion_tools(self) -> list[Tool]:
        """
        Return a list of all completion tools available for the task.
        """
        tools = []
        completion_tools = self.completion_tools
        if completion_tools is None:
            completion_tools = ["SUCCEED", "FAIL"]

        if "SUCCEED" in completion_tools:
            tools.append(self.get_success_tool())
        if "FAIL" in completion_tools:
            tools.append(self.get_fail_tool())
        return tools

    def get_prompt(self) -> str:
        """
        Generate a prompt to share information about the task with an agent.
        """
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.TaskTemplate(template=self.prompt, task=self)
        return template.render()

    def set_status(self, status: TaskStatus):
        self.status = status

        # update TUI
        if tui := ctx.get("tui"):
            tui.update_task(self)

    def mark_running(self):
        """Mark the task as running and emit a TaskStart event."""
        self.set_status(TaskStatus.RUNNING)
        if orchestrator := ctx.get("orchestrator"):
            from controlflow.events.task_events import TaskStart

            orchestrator.add_event(TaskStart(task=self))

    def mark_successful(self, result: T = None):
        """Mark the task as successful and emit a TaskSuccess event."""
        self.result = self.validate_result(result)
        self.set_status(TaskStatus.SUCCESSFUL)
        if orchestrator := ctx.get("orchestrator"):
            from controlflow.events.task_events import TaskSuccess

            orchestrator.add_event(TaskSuccess(task=self, result=result))

    def mark_failed(self, reason: Optional[str] = None):
        """Mark the task as failed and emit a TaskFailure event."""
        self.result = reason
        self.set_status(TaskStatus.FAILED)
        if orchestrator := ctx.get("orchestrator"):
            from controlflow.events.task_events import TaskFailure

            orchestrator.add_event(TaskFailure(task=self, reason=reason))

    def mark_skipped(self):
        """Mark the task as skipped and emit a TaskSkipped event."""
        self.set_status(TaskStatus.SKIPPED)
        if orchestrator := ctx.get("orchestrator"):
            from controlflow.events.task_events import TaskSkipped

            orchestrator.add_event(TaskSkipped(task=self))

    def get_success_tool(self) -> Tool:
        """
        Create an agent-compatible tool for marking this task as successful.
        """
        options = {}
        instructions = []
        metadata = {
            "is_completion_tool": True,
            "is_success_tool": True,
            "completion_task": self,
        }
        result_schema = None

        # if the result_type is a tuple of options, then we want the LLM to provide
        # a single integer index instead of writing out the entire option. Therefore
        # we create a tool that describes a series of options and accepts the index
        # as a result.
        if isinstance(self.result_type, Labels):
            result_schema = int
            options = {}
            serialized_options = {}
            for i, option in enumerate(self.result_type):
                options[i] = option
                try:
                    serialized = TypeAdapter(type(option)).dump_python(option)
                except PydanticSchemaGenerationError:
                    serialized = repr(option)
                serialized_options[i] = serialized
            options_str = "\n\n".join(
                f"Option {i}: {option}" for i, option in serialized_options.items()
            )
            instructions.append(
                unwrap(
                    """
                    Provide a single integer as the task result, corresponding to the index
                    of your chosen option. Your options are: 
                    
                    {options_str}
                    """
                ).format(options_str=options_str)
            )

        # otherwise try to load the schema for the result type
        elif self.result_type is not None:
            try:
                # see if the result type is a valid pydantic type
                TypeAdapter(self.result_type)
                result_schema = self.result_type
            except PydanticSchemaGenerationError:
                pass
            if result_schema is None:
                raise ValueError(
                    f"Could not load or infer schema for result type {self.result_type}. "
                    "Please use a custom type or add compatibility."
                )

        # for basemodel subclasses, we accept the model properties directly as kwargs
        if safe_issubclass(result_schema, BaseModel):
            instructions.append(
                unwrap(
                    f"""
                    Use this tool to mark the task as successful and provide a
                    result. The result schema is: {result_schema}
                    """
                )
            )

            def succeed(**kwargs) -> str:
                self.mark_successful(result=result_schema(**kwargs))
                return f"{self.friendly_name()} marked successful."

            return Tool(
                fn=succeed,
                name=f"mark_task_{self.id}_successful",
                description=f"Mark task {self.id} as successful.",
                instructions="\n\n".join(instructions) or None,
                parameters=result_schema.model_json_schema(),
                metadata=metadata,
            )

        # for all other results, we create a single `result` kwarg to capture the result
        elif result_schema is not None:
            instructions.append(
                unwrap(
                    f"""
                    Use this tool to mark the task as successful and provide a
                    `result` value. The `result` value has the following schema:
                    {result_schema}.
                    """
                )
            )

            @tool(
                name=f"mark_task_{self.id}_successful",
                description=f"Mark task {self.id} as successful.",
                instructions="\n\n".join(instructions) or None,
                include_return_description=False,
                metadata=metadata,
            )
            def succeed(result: result_schema) -> str:  # type: ignore
                if self.is_successful():
                    raise ValueError(
                        f"{self.friendly_name()} is already marked successful."
                    )
                if options:
                    if result not in options:
                        raise ValueError(
                            f"Invalid option. Please choose one of {options}"
                        )
                    result = options[result]
                self.mark_successful(result=result)
                return f"{self.friendly_name()} marked successful."

            return succeed
        # for no result schema, we provide a tool that takes no arguments
        else:

            @tool(
                name=f"mark_task_{self.id}_successful",
                description=f"Mark task {self.id} as successful.",
                instructions="\n\n".join(instructions) or None,
                include_return_description=False,
                metadata=metadata,
            )
            def succeed() -> str:
                self.mark_successful()
                return f"{self.friendly_name()} marked successful."

        return succeed

    def get_fail_tool(self) -> Tool:
        """
        Create an agent-compatible tool for failing this task.
        """

        @tool(
            name=f"mark_task_{self.id}_failed",
            description=unwrap(
                f"""Mark task {self.id} as failed. Only use when technical
                 errors prevent success. Provide a detailed reason for the
                 failure."""
            ),
            include_return_description=False,
            metadata={
                "is_completion_tool": True,
                "is_fail_tool": True,
                "completion_task": self,
            },
        )
        def fail(reason: str) -> str:
            self.mark_failed(reason=reason)
            return f"{self.friendly_name()} marked failed."

        return fail

    def validate_result(self, raw_result: Any) -> T:
        if self.result_type is None and raw_result is not None:
            raise ValueError("Task has result_type=None, but a result was provided.")
        elif isinstance(self.result_type, Labels):
            if raw_result not in self.result_type:
                raise ValueError(
                    f"Result {raw_result} is not in the list of valid result types: {self.result_type}"
                )
            else:
                result = raw_result
        elif self.result_type is not None:
            try:
                result = TypeAdapter(self.result_type).validate_python(raw_result)
            except PydanticSchemaGenerationError:
                if isinstance(raw_result, dict):
                    result = self.result_type(**raw_result)
                else:
                    result = self.result_type(raw_result)

        # the raw result is None
        else:
            result = raw_result

            # Convert DataFrame schema back into pd.DataFrame object
            # if result_type == PandasDataFrame:
            #     import pandas as pd

            #     result = pd.DataFrame(**result)
            # elif result_type == PandasSeries:
            #     import pandas as pd

            #     result = pd.Series(**result)

        # apply custom validation
        if self.result_validator is not None:
            result = self.result_validator(result)

        return result


def _generate_result_schema(result_type: type[T]) -> type[T]:
    if result_type is None:
        return None

    result_schema = None
    # try loading pydantic-compatible schemas
    try:
        TypeAdapter(result_type)
        result_schema = result_type
    except PydanticSchemaGenerationError:
        pass
    if result_schema is None:
        raise ValueError(
            f"Could not load or infer schema for result type {result_type}. "
            "Please use a custom type or add compatibility."
        )
    return result_schema
