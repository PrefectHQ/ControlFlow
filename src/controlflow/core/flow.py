from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union

from marvin.beta.assistants import Thread
from openai.types.beta.threads import Message
from pydantic import Field, field_validator

import controlflow
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.types import ControlFlowModel, ToolType

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.task import Task
logger = get_logger(__name__)


class Flow(ControlFlowModel):
    name: Optional[str] = None
    description: Optional[str] = None
    thread: Thread = Field(None, validate_default=True)
    tools: list[ToolType] = Field(
        default_factory=list,
        description="Tools that will be available to every agent in the flow",
    )
    agents: list["Agent"] = Field(
        description="The default agents for the flow. These agents will be used "
        "for any task that does not specify agents.",
        default_factory=list,
    )
    _tasks: dict[str, "Task"] = {}
    context: dict[str, Any] = {}

    @field_validator("thread", mode="before")
    def _load_thread_from_ctx(cls, v):
        if v is None:
            v = ctx.get("thread", None)
            if v is None:
                v = Thread()

        return v

    def add_task(self, task: "Task"):
        if self._tasks.get(task.id, task) is not task:
            raise ValueError(
                f"A different task with id '{task.id}' already exists in flow."
            )
        self._tasks[task.id] = task

    @contextmanager
    def _context(self):
        with ctx(flow=self, tasks=[]):
            yield self

    def __enter__(self):
        self.__cm = self._context()
        return self.__cm.__enter__()

    def __exit__(self, *exc_info):
        return self.__cm.__exit__(*exc_info)


GLOBAL_FLOW = None


def get_flow() -> Flow:
    """
    Loads the flow from the context.
    """
    flow: Union[Flow, None] = ctx.get("flow")
    if not flow:
        if controlflow.settings.enable_global_flow:
            return GLOBAL_FLOW
        else:
            raise ValueError("No flow found in context and global flow is disabled.")
    return flow


def reset_global_flow():
    global GLOBAL_FLOW
    GLOBAL_FLOW = Flow()


def get_flow_messages(limit: int = None) -> list[Message]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    return flow.thread.get_messages(limit=limit)
