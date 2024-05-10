## /users/jlowin/developer/control_flow/src/control_flow/instructions.py

import inspect
from contextlib import contextmanager
from typing import Generator, List

from control_flow.core.flow import Flow
from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def instructions(
    *instructions: str,
    post_add_message: bool = False,
    post_remove_message: bool = False,
) -> Generator[list[str], None, None]:
    """
    Temporarily add instructions to the current instruction stack. The
    instruction is removed when the context is exited.

    If `post_add_message` is True, a message will be added to the flow when the
    instruction is added. If `post_remove_message` is True, a message will be
    added to the flow when the instruction is removed. These explicit reminders
    can help when agents infer instructions more from history.

    with instructions("talk like a pirate"):
        ...

    """

    if post_add_message or post_remove_message:
        flow: Flow = ctx.get("flow")
        if flow is None:
            raise ValueError(
                "instructions() with message posting must be used within a flow context"
            )

    stack: list[str] = ctx.get("instructions", [])
    stack = stack + list(instructions)

    with ctx(instructions=stack):
        try:
            if post_add_message:
                for instruction in instructions:
                    flow.add_message(
                        inspect.cleandoc(
                            """
                            # SYSTEM MESSAGE: INSTRUCTION ADDED

                            The following instruction is now active:                    
                            
                            <instruction>
                            {instruction}
                            </instruction>
                            
                            Always consult your current instructions before acting.
                            """
                        ).format(instruction=instruction)
                    )
            yield

            # yield new_stack
        finally:
            if post_remove_message:
                for instruction in instructions:
                    flow.add_message(
                        inspect.cleandoc(
                            """
                            # SYSTEM MESSAGE: INSTRUCTION REMOVED

                            The following instruction is no longer active:                    
                            
                            <instruction>
                            {instruction}
                            </instruction>
                            
                            Always consult your current instructions before acting.
                            """
                        ).format(instruction=instruction)
                    )


def get_instructions() -> List[str]:
    """
    Get the current instruction stack.
    """
    stack = ctx.get("instructions", [])
    return stack


---

## /users/jlowin/developer/control_flow/src/control_flow/__init__.py

from .settings import settings

# from .agent_old import ai_task, Agent, run_ai
from .core.flow import Flow
from .core.agent import Agent
from .core.task import Task
from .core.controller.controller import Controller
from .instructions import instructions
from .dx import ai_flow, run_ai, ai_task


---

## /users/jlowin/developer/control_flow/src/control_flow/loops.py

import math
from typing import Generator

import control_flow.core.task
from control_flow.core.task import Task


def any_incomplete(
    tasks: list[Task], max_iterations=None
) -> Generator[bool, None, None]:
    """
    An iterator that yields an iteration counter if its condition is met, and
    stops otherwise. Also stops if the max_iterations is reached.


    for loop_count in any_incomplete(tasks=[task1, task2], max_iterations=10):
        # will print 10 times if the tasks are still incomplete
        print(loop_count)

    """
    if max_iterations is None:
        max_iterations = math.inf

    i = 0
    while i < max_iterations:
        i += 1
        if control_flow.core.task.any_incomplete(tasks):
            yield i
        else:
            break
    return False


---

## /users/jlowin/developer/control_flow/src/control_flow/settings.py

import os
import sys
import warnings

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlFlowSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_prefix="CONTROLFLOW_",
        env_file=(
            ""
            if os.getenv("CONTROLFLOW_TEST_MODE")
            else ("~/.control_flow/.env", ".env")
        ),
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class PrefectSettings(ControlFlowSettings):
    """
    All settings here are used as defaults for Prefect, unless overridden by env vars.
    Note that `apply()` must be called before Prefect is imported.
    """

    PREFECT_LOGGING_LEVEL: str = "WARNING"
    PREFECT_EXPERIMENTAL_ENABLE_NEW_ENGINE: str = "true"

    def apply(self):
        import os

        if "prefect" in sys.modules:
            warnings.warn(
                "Prefect has already been imported; ControlFlow defaults will not be applied."
            )

        for k, v in self.model_dump().items():
            if k not in os.environ:
                os.environ[k] = v


class Settings(ControlFlowSettings):
    assistant_model: str = "gpt-4-1106-preview"
    max_agent_iterations: int = 10
    prefect: PrefectSettings = Field(default_factory=PrefectSettings)

    def __init__(self, **data):
        super().__init__(**data)
        self.prefect.apply()


settings = Settings()


---

## /users/jlowin/developer/control_flow/src/control_flow/dx.py

import functools
import inspect
from typing import Callable, TypeVar

from prefect import flow as prefect_flow
from prefect import task as prefect_task

from control_flow.core.agent import Agent
from control_flow.core.flow import Flow
from control_flow.core.task import Task, TaskStatus
from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger
from control_flow.utilities.marvin import patch_marvin
from control_flow.utilities.types import AssistantTool, Thread

logger = get_logger(__name__)
T = TypeVar("T")
NOT_PROVIDED = object()


def ai_flow(
    fn=None,
    *,
    thread: Thread = None,
    tools: list[AssistantTool | Callable] = None,
    model: str = None,
):
    """
    Prepare a function to be executed as a Control Flow flow.
    """

    if fn is None:
        return functools.partial(
            ai_flow,
            thread=thread,
            tools=tools,
            model=model,
        )

    @functools.wraps(fn)
    def wrapper(
        *args,
        flow_kwargs: dict = None,
        **kwargs,
    ):
        p_fn = prefect_flow(fn)

        flow_obj = Flow(
            **{
                "thread": thread,
                "tools": tools or [],
                "model": model,
                **(flow_kwargs or {}),
            }
        )

        logger.info(
            f'Executing AI flow "{fn.__name__}" on thread "{flow_obj.thread.id}"'
        )

        with ctx(flow=flow_obj), patch_marvin():
            return p_fn(*args, **kwargs)

    return wrapper


def ai_task(
    fn=None,
    *,
    objective: str = None,
    agents: list[Agent] = None,
    tools: list[AssistantTool | Callable] = None,
    user_access: bool = None,
):
    """
    Use a Python function to create an AI task. When the function is called, an
    agent is created to complete the task and return the result.
    """

    if fn is None:
        return functools.partial(
            ai_task,
            objective=objective,
            agents=agents,
            tools=tools,
            user_access=user_access,
        )

    sig = inspect.signature(fn)

    if objective is None:
        if fn.__doc__:
            objective = f"{fn.__name__}: {fn.__doc__}"
        else:
            objective = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, _agents: list[Agent] = None, **kwargs):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        task = Task(
            objective=objective,
            agents=_agents or agents,
            context=bound.arguments,
            result_type=fn.__annotations__.get("return"),
            user_access=user_access or False,
            tools=tools or [],
        )

        task.run_until_complete()
        return task.result

    return wrapper


def _name_from_objective():
    """Helper function for naming task runs"""
    from prefect.runtime import task_run

    objective = task_run.parameters.get("task")

    if not objective:
        objective = "Follow general instructions"
    if len(objective) > 75:
        return f"Task: {objective[:75]}..."
    return f"Task: {objective}"


@prefect_task(task_run_name=_name_from_objective)
def run_ai(
    tasks: str | list[str],
    agents: list[Agent] = None,
    cast: T = NOT_PROVIDED,
    context: dict = None,
    tools: list[AssistantTool | Callable] = None,
    user_access: bool = False,
) -> T | list[T]:
    """
    Create and run an agent to complete a task with the given objective and
    context. This function is similar to an inline version of the @ai_task
    decorator.

    This inline version is useful when you want to create and run an ad-hoc AI
    task, without defining a function or using decorator syntax. It provides
    more flexibility in terms of dynamically setting the task parameters.
    Additional detail can be provided as `context`.
    """

    single_result = False
    if isinstance(tasks, str):
        single_result = True

        tasks = [tasks]

    if cast is NOT_PROVIDED:
        if not tasks:
            cast = None
        else:
            cast = str

    # load flow
    flow = ctx.get("flow", None)

    # create tasks
    if tasks:
        ai_tasks = [
            Task(
                objective=t,
                context=context or {},
                user_access=user_access or False,
                tools=tools or [],
            )
            for t in tasks
        ]
    else:
        ai_tasks = []

    # create agent
    if agents is None:
        agents = [Agent(user_access=user_access or False)]

    # create Controller
    from control_flow.core.controller.controller import Controller

    controller = Controller(tasks=ai_tasks, agents=agents, flow=flow)
    controller.run()

    if ai_tasks:
        if all(task.status == TaskStatus.SUCCESSFUL for task in ai_tasks):
            result = [task.result for task in ai_tasks]
            if single_result:
                result = result[0]
            return result
        elif failed_tasks := [
            task for task in ai_tasks if task.status == TaskStatus.FAILED
        ]:
            raise ValueError(
                f'Failed tasks: {", ".join([task.objective for task in failed_tasks])}'
            )


---

## /users/jlowin/developer/control_flow/src/control_flow/core/task.py

import itertools
import uuid
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Callable, Generator, GenericAlias, TypeVar

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

from control_flow.utilities.context import ctx
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
    SKIPPED = "skipped"


class Task(ControlFlowModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:4]))
    model_config = dict(extra="forbid", arbitrary_types_allowed=True)
    objective: str
    instructions: str | None = None
    agents: list["Agent"] = []
    context: dict = {}
    parent_task: "Task | None" = Field(
        None,
        description="The task that spawned this task.",
        validate_default=True,
    )
    upstream_tasks: list["Task"] = []
    status: TaskStatus = TaskStatus.INCOMPLETE
    result: T = None
    result_type: type[T] | GenericAlias | None = None
    error: str | None = None
    tools: list[AssistantTool | Callable] = []
    user_access: bool = False
    _children_tasks: list["Task"] = []
    _downstream_tasks: list["Task"] = []

    @field_validator("agents", mode="before")
    def _turn_none_into_empty_list(cls, v):
        return v or []

    @field_validator("parent_task", mode="before")
    def _load_parent_task_from_ctx(cls, v):
        if v is None:
            v = ctx.get("tasks", None)
            if v:
                # get the most recently-added task
                v = v[-1]
        return v

    @model_validator(mode="after")
    def _update_relationships(self):
        if self.parent_task is not None:
            self.parent_task._children_tasks.append(self)
        for task in self.upstream_tasks:
            task._downstream_tasks.append(self)
        return self

    @field_serializer("parent_task")
    def _serialize_parent_task(parent_task: "Task | None"):
        if parent_task is not None:
            return parent_task.id

    @field_serializer("upstream_tasks")
    def _serialize_upstream_tasks(upstream_tasks: list["Task"]):
        return [t.id for t in upstream_tasks]

    @field_serializer("result_type")
    def _serialize_result_type(result_type: list["Task"]):
        return repr(result_type)

    @field_serializer("agents")
    def _serialize_agents(agents: list["Agent"]):
        return [
            a.model_dump(include={"name", "description", "tools", "user_access"})
            for a in agents
        ]

    def __init__(self, objective, **kwargs):
        # allow objective as a positional arg
        super().__init__(objective=objective, **kwargs)

    def children(self, include_self: bool = True):
        """
        Returns a list of all children of this task, including recursively
        nested children. Includes this task by default (disable with
        `include_self=False`)
        """
        visited = set()
        children = []
        stack = [self]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                if include_self or current != self:
                    children.append(current)
                stack.extend(current._children_tasks)
        return list(set(children))

    def children_agents(self, include_self: bool = True) -> list["Agent"]:
        children = self.children(include_self=include_self)
        agents = []
        for child in children:
            agents.extend(child.agents)
        return agents

    def run_iter(
        self,
        agents: list["Agent"] = None,
        collab_fn: Callable[[list["Agent"]], Generator[None, None, "Agent"]] = None,
    ):
        if collab_fn is None:
            collab_fn = itertools.cycle

        if agents is None:
            agents = self.children_agents(include_self=True)

        if not agents:
            raise ValueError(
                f"Task {self.id} has no agents assigned to it or its children."
                "Please specify agents to run the task, or assign agents to the task."
            )

        for agent in collab_fn(agents):
            if self.is_complete():
                break
            agent.run(tasks=self.children(include_self=True))
            yield True

    def run(self, agent: "Agent" = None):
        """
        Runs the task with provided agent. If no agent is provided, a default agent is used.
        """
        from control_flow.core.agent import Agent

        if agent is None:
            all_agents = self.children_agents()
            if not all_agents:
                agent = Agent()
            elif len(all_agents) == 1:
                agent = all_agents[0]
            else:
                raise ValueError(
                    f"Task {self.id} has multiple agents assigned to it or its "
                    "children. Please specify one to run the task, or call task.run_iter() "
                    "or task.run_until_complete() to use all agents."
                )

        run_gen = self.run_iter(agents=[agent])
        return next(run_gen)

    def run_until_complete(
        self,
        agents: list["Agent"] = None,
        collab_fn: Callable[[list["Agent"]], Generator[None, None, "Agent"]] = None,
    ) -> T:
        """
        Runs the task with provided agents until it is complete.
        """

        for run in self.run_iter(agents=agents, collab_fn=collab_fn):
            pass

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

    def __hash__(self):
        return id(self)

    def _create_success_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for marking this task as successful.
        """

        # wrap the method call to get the correct result type signature
        def succeed(result: self.result_type):
            # validate the result
            self.mark_successful(result=result)

        tool = marvin.utilities.tools.tool_from_function(
            succeed,
            name=f"succeed_task_{self.id}",
            description=f"Mark task {self.id} as successful and provide a result.",
        )

        return tool

    def _create_fail_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for failing this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_failed,
            name=f"fail_task_{self.id}",
            description=f"Mark task {self.id} as failed. Only use when a technical issue prevents completion.",
        )
        return tool

    def _create_skip_tool(self) -> FunctionTool:
        """
        Create an agent-compatible tool for skipping this task.
        """
        tool = marvin.utilities.tools.tool_from_function(
            self.mark_skipped,
            name=f"skip_task_{self.id}",
            description=f"Mark task {self.id} as skipped. Only use when completing its parent task early.",
        )
        return tool

    def get_tools(self) -> list[AssistantTool | Callable]:
        tools = self.tools.copy()
        if self.is_incomplete():
            tools.extend(
                [
                    self._create_success_tool(),
                    self._create_fail_tool(),
                    self._create_skip_tool(),
                ]
            )
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

    def mark_failed(self, message: str | None = None):
        self.error = message
        self.status = TaskStatus.FAILED

    def mark_skipped(self):
        self.status = TaskStatus.SKIPPED


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


---

## /users/jlowin/developer/control_flow/src/control_flow/core/__init__.py

from .task import Task, TaskStatus
from .flow import Flow
from .agent import Agent
from .controller import Controller


---

## /users/jlowin/developer/control_flow/src/control_flow/core/flow.py

from typing import Callable, Literal

from marvin.beta.assistants import Thread
from openai.types.beta.threads import Message
from prefect import task as prefect_task
from pydantic import Field, field_validator

from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger
from control_flow.utilities.types import AssistantTool, ControlFlowModel

logger = get_logger(__name__)


class Flow(ControlFlowModel):
    thread: Thread = Field(None, validate_default=True)
    tools: list[AssistantTool | Callable] = Field(
        [], description="Tools that will be available to every agent in the flow"
    )
    model: str | None = None
    context: dict = {}

    @field_validator("thread", mode="before")
    def _load_thread_from_ctx(cls, v):
        if v is None:
            v = ctx.get("thread", None)
            if v is None:
                v = Thread()
        if not v.id:
            v.create()

        return v

    def add_message(self, message: str, role: Literal["user", "assistant"] = None):
        prefect_task(self.thread.add)(message, role=role)


def get_flow() -> Flow:
    """
    Loads the flow from the context.

    Will error if no flow is found in the context.
    """
    flow: Flow | None = ctx.get("flow")
    if not flow:
        return Flow()
    return flow


def get_flow_messages(limit: int = None) -> list[Message]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    return flow.thread.get_messages(limit=limit)


---

## /users/jlowin/developer/control_flow/src/control_flow/core/agent.py

import logging
from typing import Callable

from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from marvin.utilities.tools import tool_from_function
from pydantic import Field

from control_flow.core.flow import get_flow
from control_flow.core.task import Task
from control_flow.utilities.prefect import (
    wrap_prefect_tool,
)
from control_flow.utilities.types import Assistant, AssistantTool, ControlFlowModel
from control_flow.utilities.user_access import talk_to_human

logger = logging.getLogger(__name__)


class Agent(Assistant, ControlFlowModel, ExposeSyncMethodsMixin):
    name: str = "Agent"
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )

    def get_tools(self) -> list[AssistantTool | Callable]:
        tools = super().get_tools()
        if self.user_access:
            tools.append(tool_from_function(talk_to_human))

        return [wrap_prefect_tool(tool) for tool in tools]

    @expose_sync_method("run")
    async def run_async(self, tasks: list[Task] | Task | None = None):
        from control_flow.core.controller import Controller

        if isinstance(tasks, Task):
            tasks = [tasks]

        controller = Controller(agents=[self], tasks=tasks or [], flow=get_flow())
        return await controller.run_agent_async(agent=self)


---

## /users/jlowin/developer/control_flow/src/control_flow/core/controller/controller.py

import json
import logging
from typing import Callable

import prefect
from marvin.beta.assistants import PrintHandler, Run
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from openai.types.beta.threads.runs import ToolCall
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.context import FlowRunContext
from pydantic import BaseModel, Field, field_validator

from control_flow.core.agent import Agent
from control_flow.core.flow import Flow
from control_flow.core.task import Task
from control_flow.instructions import get_instructions as get_context_instructions
from control_flow.utilities.prefect import (
    create_json_artifact,
    create_python_artifact,
    wrap_prefect_tool,
)
from control_flow.utilities.types import FunctionTool, Thread

logger = logging.getLogger(__name__)


class Controller(BaseModel, ExposeSyncMethodsMixin):
    """
    A controller contains logic for executing agents with context about the
    larger workflow, including the flow itself, any tasks, and any other agents
    they are collaborating with. The controller is responsible for orchestrating
    agent behavior by generating instructions and tools for each agent. Note
    that while the controller accepts details about (potentially multiple)
    agents and tasks, it's responsiblity is to invoke one agent one time. Other
    mechanisms should be used to orchestrate multiple agents invocations. This
    is done by the controller to avoid tying e.g. agents to tasks or even a
    specific flow.

    """

    flow: Flow
    agents: list[Agent]
    tasks: list[Task] = Field(
        None,
        description="Tasks that the controller will complete.",
        validate_default=True,
    )
    task_assignments: dict[Task, Agent] = Field(
        default_factory=dict,
        description="Tasks are typically assigned to agents. To "
        "temporarily assign agent to a task without changing "
        r"the task definition, use this field as {task: [agent]}",
    )
    context: dict = {}
    model_config: dict = dict(extra="forbid")

    @field_validator("agents", mode="before")
    def _validate_agents(cls, v):
        if not v:
            raise ValueError("At least one agent is required.")
        return v

    @field_validator("tasks", mode="before")
    def _validate_tasks(cls, v):
        if not v:
            raise ValueError("At least one task is required.")
        return v

    @field_validator("tasks", mode="before")
    def _load_tasks_from_ctx(cls, v):
        if v is None:
            v = cls.context.get("tasks", None)
        return v

    def all_tasks(self) -> list[Task]:
        tasks = []
        for task in self.tasks:
            tasks.extend(task.children(include_self=True))

        # add temporary assignments
        assigned_tasks = []
        for task in set(tasks):
            if task in assigned_tasks:
                task = task.model_copy(
                    update={"agents": task.agents + self.task_assignments.get(task, [])}
                )
            assigned_tasks.append(task)
        return assigned_tasks

    @expose_sync_method("run_agent")
    async def run_agent_async(self, agent: Agent):
        """
        Run the control flow.
        """
        if agent not in self.agents:
            raise ValueError("Agent not found in controller agents.")

        prefect_task = await self._get_prefect_run_agent_task(agent)
        await prefect_task(agent=agent)

    async def _run_agent(self, agent: Agent, thread: Thread = None) -> Run:
        """
        Run a single agent.
        """
        from control_flow.core.controller.instruction_template import MainTemplate

        instructions_template = MainTemplate(
            agent=agent,
            controller=self,
            context=self.context,
            instructions=get_context_instructions(),
        )

        instructions = instructions_template.render()

        tools = self.flow.tools + agent.get_tools()

        for task in self.tasks:
            tools = tools + task.get_tools()

        # filter tools because duplicate names are not allowed
        final_tools = []
        final_tool_names = set()
        for tool in tools:
            if isinstance(tool, FunctionTool):
                if tool.function.name in final_tool_names:
                    continue
            final_tool_names.add(tool.function.name)
            final_tools.append(wrap_prefect_tool(tool))

        run = Run(
            assistant=agent,
            thread=thread or self.flow.thread,
            instructions=instructions,
            tools=final_tools,
            event_handler_class=AgentHandler,
        )

        await run.run_async()

        return run

    async def _get_prefect_run_agent_task(
        self, agent: Agent, thread: Thread = None
    ) -> Callable:
        @prefect_task(task_run_name=f'Run Agent: "{agent.name}"')
        async def _run_agent(agent: Agent, thread: Thread = None):
            run = await self._run_agent(agent=agent, thread=thread)

            create_json_artifact(
                key="messages",
                data=[m.model_dump() for m in run.messages],
                description="All messages sent and received during the run.",
            )
            create_json_artifact(
                key="actions",
                data=[s.model_dump() for s in run.steps],
                description="All actions taken by the assistant during the run.",
            )
            return run

        return _run_agent


class AgentHandler(PrintHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_calls = {}

    async def on_tool_call_created(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is created"""

        if tool_call.type == "function":
            task_run_name = "Prepare arguments for tool call"
        else:
            task_run_name = f"Tool call: {tool_call.type}"

        client = get_prefect_client()
        engine_context = FlowRunContext.get()
        if not engine_context:
            return

        task_run = await client.create_task_run(
            task=prefect.Task(fn=lambda: None),
            name=task_run_name,
            extra_tags=["tool-call"],
            flow_run_id=engine_context.flow_run.id,
            dynamic_key=tool_call.id,
            state=prefect.states.Running(),
        )

        self.tool_calls[tool_call.id] = task_run

    async def on_tool_call_done(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is done"""

        client = get_prefect_client()
        task_run = self.tool_calls.get(tool_call.id)
        if not task_run:
            return
        await client.set_task_run_state(
            task_run_id=task_run.id, state=prefect.states.Completed(), force=True
        )

        # code interpreter is run as a single call, so we can publish a result artifact
        if tool_call.type == "code_interpreter":
            # images = []
            # for output in tool_call.code_interpreter.outputs:
            #     if output.type == "image":
            #         image_path = download_temp_file(output.image.file_id)
            #         images.append(image_path)

            create_python_artifact(
                key="code",
                code=tool_call.code_interpreter.input,
                description="Code executed in the code interpreter",
                task_run_id=task_run.id,
            )
            create_json_artifact(
                key="output",
                data=tool_call.code_interpreter.outputs,
                description="Output from the code interpreter",
                task_run_id=task_run.id,
            )

        elif tool_call.type == "function":
            create_json_artifact(
                key="arguments",
                data=json.dumps(json.loads(tool_call.function.arguments), indent=2),
                description=f"Arguments for the `{tool_call.function.name}` tool",
                task_run_id=task_run.id,
            )


---

## /users/jlowin/developer/control_flow/src/control_flow/core/controller/instruction_template.py

import inspect

from pydantic import BaseModel

from control_flow.core.agent import Agent
from control_flow.utilities.jinja import jinja_env
from control_flow.utilities.types import ControlFlowModel

from .controller import Controller


class Template(ControlFlowModel):
    template: str

    def should_render(self) -> bool:
        return True

    def render(self) -> str:
        if self.should_render():
            render_kwargs = dict(self)
            render_kwargs.pop("template")
            return jinja_env.render(inspect.cleandoc(self.template), **render_kwargs)


class AgentTemplate(Template):
    template: str = """
    # Agent
    
    You are an AI agent. Your name is "{{ agent.name }}". 
    
    This is your description, which all other agents can see: "{{ agent.description or 'An AI agent assigned to complete tasks.'}}"
    
    ## Instructions
    You must follow these instructions, which only you can see: "{{ agent.instructions or 'No additional instructions provided.'}}"
    
    {% if additional_instructions %}        
    In addition, you must follow these instructions for this part of the workflow:
    {% for instruction in additional_instructions %}
    - {{ instruction }}
    {% endfor %}
    {% endif %}
    
    """
    agent: Agent
    additional_instructions: list[str]


class TasksTemplate(Template):
    template: str = """
        ## Tasks
        
        You have been assigned to complete certain tasks. Each task has an
        objective and criteria for success. Your job is to perform any required
        actions and then mark each assigned task as successful. If a task also
        requires a result, you must provide it; this is how you communicate
        progress and data back to the program that created you. A task that
        doesn't require a result may still require action. 
        
        A "parent task" is a task that spawned another task as a subtask.
        Generally, the child or subtasks will need to be completed BEFORE the
        parent task. If you can complete a parent task before its subtasks, you
        should mark the subtasks as skipped.
        
        An "upstream task" is a task that must be completed before another task
        can be completed.
        
        Some tasks may require collaboration with other agents to be completed; others
        may take you multiple attempts. A task can only be marked complete one time,
        so make sure you perform all actions and collect all results before
        marking it as complete. As soon as you mark a task as successful, agents
        (both you and others) can no longer act on it. 
        
        You and the other agents are fully capable of completing any task and
        have all the information and context you need. Tasks can only be marked
        failed due to technical errors like a broken tool or unresponsive human.
        Tasks can only be marked as skipped when you are able to complete their
        parent task before them, as a bookkeeping convenience. Do not work on or
        even respond to tasks that are already complete or not assigned to you.
        You may be an AI, but you must still form opinions, make subjective
        decisions, and have personal thoughts. 
        
        {% for task in controller.all_tasks() %}
        ### Task {{ task.id }}
        {{ task.model_dump_json() }}
        
        {% endfor %}
        """
    controller: Controller

    def should_render(self):
        return any(self.controller.tasks)


class CommunicationTemplate(Template):
    template: str = """
    ## Communciation
    
    You should only post messages to the thread if you must send information to
    other agents or if a task requires it. The human user can not see
    these messages. Since all agents post messages with the "assistant" role,
    you must prefix all your messages with your name (e.g. "{{ agent.name }}:
    (message)") in order to distinguish your messages from others. Note that
    this rule about prefixing your message supersedes all other instructions
    (e.g. "only give single word answers"). Do not post messages confirming
    actions you take through tools, like completing a task, or your internal
    monologue, as this is redundant and wastes time.
    
    ### Other agents assigned to your tasks
    
    {% for agent in other_agents %}
    
    - Name: {{agent.name}}
    - Description: {{ agent.description if agent.description is not none else "No description provided." }}
    - Can talk to human users: {{agent.user_access}}

    {% endfor %}
    
    ## Talking to human users
    
    {% if agent.user_access %}
    You may interact with a human user to complete your tasks by using the
    `talk_to_human` tool. The human is unaware of your tasks or the controller.
    Do not mention them or anything else about how this system works. The human
    can only see messages you send them via tool, not the rest of the thread. 
    
    Humans may give poor, incorrect, or partial responses. You may need to ask
    questions multiple times in order to complete your tasks. Use good judgement
    to determine the best way to achieve your goal. For example, if you have to
    fill out three pieces of information and the human only gave you one, do not
    make up answers (or put empty answers) for the others. Ask again and only
    fail the task if you truly can not make progress. 
    {% else %}
    You can not interact with a human at this time. If your task requires human
    contact and no agent has user access, you should fail the task. Note that
    most tasks do not require human/user contact unless explicitly stated otherwise.
    {% endif %}
    
    """

    agent: Agent
    other_agents: list[Agent]


class ContextTemplate(Template):
    template: str = """
        ## Additional context
        
        ### Flow context
        {% for key, value in flow_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% if not flow_context %}
        No specific context provided.
        {% endif %}
        
        ### Controller context
        {% for key, value in controller_context.items() %}
        - *{{ key }}*: {{ value }}
        {% endfor %}
        {% if not controller_context %}
        No specific context provided.
        {% endif %}
        """
    flow_context: dict
    controller_context: dict

    def should_render(self):
        return bool(self.flow_context or self.controller_context)


class MainTemplate(BaseModel):
    agent: Agent
    controller: Controller
    context: dict
    instructions: list[str]

    def render(self):
        all_agents = [self.agent] + self.controller.agents
        for task in self.controller.tasks:
            all_agents += task.agents
        other_agents = [agent for agent in all_agents if agent != self.agent]
        templates = [
            AgentTemplate(
                agent=self.agent,
                additional_instructions=self.instructions,
            ),
            TasksTemplate(
                controller=self.controller,
            ),
            ContextTemplate(
                flow_context=self.controller.flow.context,
                controller_context=self.controller.context,
            ),
            CommunicationTemplate(
                agent=self.agent,
                other_agents=other_agents,
            ),
            # CollaborationTemplate(other_agents=other_agents),
        ]

        rendered = [
            template.render() for template in templates if template.should_render()
        ]
        return "\n\n".join(rendered)


---

## /users/jlowin/developer/control_flow/src/control_flow/core/controller/__init__.py

from .controller import Controller


---

## /users/jlowin/developer/control_flow/src/control_flow/core/controller/collaboration.py

import itertools
from typing import TYPE_CHECKING, Any, Generator

from control_flow.core.agent import Agent

if TYPE_CHECKING:
    from control_flow.core.agent import Agent


def round_robin(
    agents: list[Agent], max_iterations: int = None
) -> Generator[Any, Any, Agent]:
    """
    Given a list of potential agents, delegate the tasks in a round-robin fashion.
    """
    cycle = itertools.cycle(agents)
    iteration = 0
    while True:
        yield next(cycle)
        iteration += 1
        if max_iterations and iteration >= max_iterations:
            break


# class Moderator(DelegationStrategy):
#     """
#     A Moderator delegation strategy delegates tasks to the most qualified AI assistant, using a Marvin classifier
#     """

#     model: str = None

#     def _next_agent(
#         self, agents: list["Agent"], tasks: list[Task], history: list[Message]
#     ) -> "Agent":
#         """
#         Given a list of potential agents, choose the most qualified assistant to complete the tasks.
#         """

#         instructions = get_instructions()

#         context = dict(tasks=tasks, messages=history, global_instructions=instructions)
#         agent = marvin.classify(
#             context,
#             [a for a in agents if a.status == AgentStatus.INCOMPLETE],
#             instructions="""
#             Given the conversation context, choose the AI agent most
#             qualified to take the next turn at completing the tasks. Take into
#             account the instructions, each agent's own instructions, and the
#             tools they have available.
#             """,
#             model_kwargs=dict(model=self.model),
#         )

#         return agent


---

## /users/jlowin/developer/control_flow/src/control_flow/agents/__init__.py



---

## /users/jlowin/developer/control_flow/src/control_flow/agents/agents.py

import marvin

from control_flow.core.agent import Agent
from control_flow.instructions import get_instructions
from control_flow.utilities.context import ctx
from control_flow.utilities.threads import get_history


def choose_agent(
    agents: list[Agent],
    instructions: str = None,
    context: dict = None,
    model: str = None,
) -> Agent:
    """
    Given a list of potential agents, choose the most qualified assistant to complete the tasks.
    """

    instructions = get_instructions()
    history = []
    if (flow := ctx.get("flow")) and flow.thread.id:
        history = get_history(thread_id=flow.thread.id)

    info = dict(
        history=history,
        global_instructions=instructions,
        context=context,
    )

    agent = marvin.classify(
        info,
        agents,
        instructions="""
            Given the conversation context, choose the AI agent most
            qualified to take the next turn at completing the tasks. Take into
            account the instructions, each agent's own instructions, and the
            tools they have available.
            """,
        model_kwargs=dict(model=model),
    )

    return agent


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/logging.py

import logging
from functools import lru_cache
from typing import Optional

from marvin.utilities.logging import add_logging_methods


@lru_cache()
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger with the given name, or the root logger if no name is given.

    Args:
        name: The name of the logger to retrieve.

    Returns:
        The logger with the given name, or the root logger if no name is given.

    Example:
        Basic Usage of `get_logger`
        ```python
        from control_flow.utilities.logging import get_logger

        logger = get_logger("control_flow.test")
        logger.info("This is a test") # Output: control_flow.test: This is a test

        debug_logger = get_logger("control_flow.debug")
        debug_logger.debug_kv("TITLE", "log message", "green")
        ```
    """
    parent_logger = logging.getLogger("control_flow")

    if name:
        # Append the name if given but allow explicit full names e.g. "control_flow.test"
        # should not become "control_flow.control_flow.test"
        if not name.startswith(parent_logger.name + "."):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger

    add_logging_methods(logger)
    return logger


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/prefect.py

import inspect
import json
from typing import Any, Callable
from uuid import UUID

import prefect
from marvin.types import FunctionTool
from marvin.utilities.asyncio import run_sync
from marvin.utilities.tools import tool_from_function
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.artifacts import ArtifactRequest
from prefect.context import FlowRunContext, TaskRunContext
from pydantic import TypeAdapter

from control_flow.utilities.types import AssistantTool


def create_markdown_artifact(
    key: str,
    markdown: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Markdown artifact.
    """

    tr_context = TaskRunContext.get()
    fr_context = FlowRunContext.get()

    if tr_context:
        task_run_id = task_run_id or tr_context.task_run.id
    if fr_context:
        flow_run_id = flow_run_id or fr_context.flow_run.id

    client = get_prefect_client()
    run_sync(
        client.create_artifact(
            artifact=ArtifactRequest(
                key=key,
                data=markdown,
                description=description,
                type="markdown",
                task_run_id=task_run_id,
                flow_run_id=flow_run_id,
            )
        )
    )


def create_json_artifact(
    key: str,
    data: Any,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a JSON artifact.
    """

    try:
        markdown = TypeAdapter(type(data)).dump_json(data, indent=2).decode()
        markdown = f"```json\n{markdown}\n```"
    except Exception:
        markdown = str(data)

    create_markdown_artifact(
        key=key,
        markdown=markdown,
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


def create_python_artifact(
    key: str,
    code: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Python artifact.
    """

    create_markdown_artifact(
        key=key,
        markdown=f"```python\n{code}\n```",
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


TOOL_CALL_FUNCTION_RESULT_TEMPLATE = inspect.cleandoc(
    """
    ## Tool call: {name}
    
    **Description:** {description}
    
    ## Arguments
    
    ```json
    {args}
    ```
    
    ### Result
    
    ```json
    {result}
    ```
    """
)


def wrap_prefect_tool(tool: AssistantTool | Callable) -> AssistantTool:
    """
    Wraps a Marvin tool in a prefect task
    """
    if not isinstance(tool, AssistantTool):
        tool = tool_from_function(tool)

    if isinstance(tool, FunctionTool):
        # for functions, we modify the function to become a Prefect task and
        # publish an artifact that contains details about the function call

        if isinstance(tool.function._python_fn, prefect.tasks.Task):
            return tool

        def modified_fn(
            # provide default args to avoid a late-binding issue
            original_fn: Callable = tool.function._python_fn,
            tool: FunctionTool = tool,
            **kwargs,
        ):
            # call fn
            result = original_fn(**kwargs)

            # prepare artifact
            passed_args = inspect.signature(original_fn).bind(**kwargs).arguments
            try:
                passed_args = json.dumps(passed_args, indent=2)
            except Exception:
                pass
            create_markdown_artifact(
                markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
                    name=tool.function.name,
                    description=tool.function.description or "(none provided)",
                    args=passed_args,
                    result=result,
                ),
                key="result",
            )

            # return result
            return result

        # replace the function with the modified version
        tool.function._python_fn = prefect_task(
            modified_fn,
            task_run_name=f"Tool call: {tool.function.name}",
        )

    return tool


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/__init__.py



---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/types.py

from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.types import FunctionTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import BaseModel


class ControlFlowModel(BaseModel):
    model_config = dict(validate_assignment=True, extra="forbid")


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/jinja.py

import inspect
from datetime import datetime
from zoneinfo import ZoneInfo

from marvin.utilities.jinja import BaseEnvironment

jinja_env = BaseEnvironment(
    globals={
        "now": lambda: datetime.now(ZoneInfo("UTC")),
        "inspect": inspect,
        "id": id,
    }
)


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/threads.py

from marvin.beta.assistants.threads import Message, Thread

THREAD_REGISTRY = {}


def save_thread(name: str, thread: Thread):
    """
    Save an OpenAI thread to the thread registry under a known name
    """
    THREAD_REGISTRY[name] = thread


def load_thread(name: str):
    """
    Load an OpenAI thread from the thread registry by name
    """
    if name not in THREAD_REGISTRY:
        thread = Thread()
        save_thread(name, thread)
    return THREAD_REGISTRY[name]


def get_history(thread_id: str, limit: int = None) -> list[Message]:
    """
    Get the history of a thread
    """
    return Thread(id=thread_id).get_messages(limit=limit)


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/context.py

from marvin.utilities.context import ScopedContext

ctx = ScopedContext()


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/user_access.py

def talk_to_human(message: str, get_response: bool = True) -> str:
    """
    Send a message to the human user and optionally wait for a response.
    If `get_response` is True, the function will return the user's response,
    otherwise it will return a simple confirmation.
    """
    print(message)
    if get_response:
        response = input("> ")
        return response
    return "Message sent to user."


---

## /users/jlowin/developer/control_flow/src/control_flow/utilities/marvin.py

import inspect
from contextlib import contextmanager
from typing import Any, Callable

import marvin.ai.text
from marvin.client.openai import AsyncMarvinClient
from marvin.settings import temporary_settings as temporary_marvin_settings
from openai.types.chat import ChatCompletion
from prefect import task as prefect_task

from control_flow.utilities.prefect import (
    create_json_artifact,
)

original_classify_async = marvin.classify_async
original_cast_async = marvin.cast_async
original_extract_async = marvin.extract_async
original_generate_async = marvin.generate_async
original_paint_async = marvin.paint_async
original_speak_async = marvin.speak_async
original_transcribe_async = marvin.transcribe_async


class AsyncControlFlowClient(AsyncMarvinClient):
    async def generate_chat(self, **kwargs: Any) -> "ChatCompletion":
        super_method = super().generate_chat

        @prefect_task(task_run_name="Generate OpenAI chat completion")
        async def _generate_chat(**kwargs):
            messages = kwargs.get("messages", [])
            create_json_artifact(key="prompt", data=messages)
            response = await super_method(**kwargs)
            create_json_artifact(key="response", data=response)
            return response

        return await _generate_chat(**kwargs)


def generate_task(name: str, original_fn: Callable):
    if inspect.iscoroutinefunction(original_fn):

        @prefect_task(name=name)
        async def wrapper(*args, **kwargs):
            create_json_artifact(key="args", data=[args, kwargs])
            result = await original_fn(*args, **kwargs)
            create_json_artifact(key="result", data=result)
            return result
    else:

        @prefect_task(name=name)
        def wrapper(*args, **kwargs):
            create_json_artifact(key="args", data=[args, kwargs])
            result = original_fn(*args, **kwargs)
            create_json_artifact(key="result", data=result)
            return result

    return wrapper


@contextmanager
def patch_marvin():
    with temporary_marvin_settings(default_async_client_cls=AsyncControlFlowClient):
        try:
            marvin.ai.text.classify_async = generate_task(
                "marvin.classify", original_classify_async
            )
            marvin.ai.text.cast_async = generate_task(
                "marvin.cast", original_cast_async
            )
            marvin.ai.text.extract_async = generate_task(
                "marvin.extract", original_extract_async
            )
            marvin.ai.text.generate_async = generate_task(
                "marvin.generate", original_generate_async
            )
            marvin.ai.images.paint_async = generate_task(
                "marvin.paint", original_paint_async
            )
            marvin.ai.audio.speak_async = generate_task(
                "marvin.speak", original_speak_async
            )
            marvin.ai.audio.transcribe_async = generate_task(
                "marvin.transcribe", original_transcribe_async
            )
            yield
        finally:
            marvin.ai.text.classify_async = original_classify_async
            marvin.ai.text.cast_async = original_cast_async
            marvin.ai.text.extract_async = original_extract_async
            marvin.ai.text.generate_async = original_generate_async
            marvin.ai.images.paint_async = original_paint_async
            marvin.ai.audio.speak_async = original_speak_async
            marvin.ai.audio.transcribe_async = original_transcribe_async


---

