import logging
from enum import Enum
from typing import Callable, TypeVar

from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from pydantic import Field, field_validator

from control_flow.context import ctx
from control_flow.flow import Flow
from control_flow.task import Task, TaskStatus
from control_flow.types import Assistant, AssistantTool, ControlFlowModel

T = TypeVar("T")
logger = logging.getLogger(__name__)

NOT_PROVIDED = object()


class AgentStatus(Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


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
    return "Message sent to user"


class Agent(ControlFlowModel, ExposeSyncMethodsMixin):
    tasks: list[Task] = Field(description="Tasks that the agent will complete.")
    assistant: Assistant = Field(default_factory=Assistant)
    instructions: str = None
    tools: list[AssistantTool | Callable] = []
    context: dict = Field({}, validate_default=True)
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    controller_access: bool = Field(
        False,
        description="If True, the agent will communicate with the controller via messages.",
    )

    @field_validator("tasks", mode="before")
    def _validate_tasks(cls, v):
        if not v:
            raise ValueError("An agent must have at least one task.")
        return v

    def task_ids(self) -> list[tuple[int, Task]]:
        """
        Assign an ID to each task so they can be identified by the assistant.
        """
        return [(i + 1, task) for i, task in enumerate(self.tasks)]

    def get_tools(self) -> list[AssistantTool | Callable]:
        """
        Get all tools from the agent and its tasks.
        """
        tools = self.tools
        for i, task in self.task_ids():
            tools = tools + task.get_tools(task_id=i)
        if self.user_access:
            tools.append(talk_to_human)
        return tools

    @property
    def status(self) -> AgentStatus:
        """
        Check if all tasks have been completed.
        """
        if any(task.status == TaskStatus.PENDING for task in self.tasks):
            return AgentStatus.INCOMPLETE
        else:
            return AgentStatus.COMPLETE

    @expose_sync_method("run")
    async def run_async(self, flow: Flow = None):
        from control_flow.controller import SingleAgentController

        controller = SingleAgentController(agents=[self], flow=flow)
        return await controller.run()


# @prefect_task(task_run_name=_name_from_objective)
def run_ai_task(
    task: str = None,
    cast: T = NOT_PROVIDED,
    context: dict = None,
    user_access: bool = None,
    **agent_kwargs: dict,
) -> T:
    """
    Create and run an agent to complete a task with the given objective and
    context. This function is similar to an inline version of the @ai_task
    decorator.

    This inline version is useful when you want to create and run an ad-hoc AI
    task, without defining a function or using decorator syntax. It provides
    more flexibility in terms of dynamically setting the task parameters.
    Additional detail can be provided as `context`.
    """

    if cast is NOT_PROVIDED:
        if not task:
            cast = None
        else:
            cast = str

    # load flow
    flow = ctx.get("flow", None)

    # create tasks
    if task:
        ai_tasks = [Task[cast](objective=task, context=context or {})]
    else:
        ai_tasks = []

    # run agent
    agent = Agent(tasks=ai_tasks, user_access=user_access or False, **agent_kwargs)
    agent.run(flow=flow)

    if ai_tasks:
        if ai_tasks[0].status == TaskStatus.COMPLETED:
            return ai_tasks[0].result
        elif ai_tasks[0].status == TaskStatus.FAILED:
            raise ValueError(ai_tasks[0].error)
