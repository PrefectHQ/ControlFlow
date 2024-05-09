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
