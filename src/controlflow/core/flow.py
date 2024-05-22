import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from pydantic import Field

from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.types import ControlFlowModel, MessageType

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.task import Task
logger = get_logger(__name__)


class Flow(ControlFlowModel):
    name: Optional[str] = None
    description: Optional[str] = None
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tools: list[Callable] = Field(
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__cm_stack = []

    def add_task(self, task: "Task"):
        if self._tasks.get(task.id, task) is not task:
            raise ValueError(
                f"A different task with id '{task.id}' already exists in flow."
            )
        self._tasks[task.id] = task

    @contextmanager
    def _context(self):
        with ctx(flow=self):
            yield self

    def __enter__(self):
        # use stack so we can enter the context multiple times
        self.__cm_stack.append(self._context())
        return self.__cm_stack[-1].__enter__()

    def __exit__(self, *exc_info):
        return self.__cm_stack.pop().__exit__(*exc_info)

    def run(self):
        """
        Runs the flow.
        """
        from controlflow.core.controller import Controller

        if self._tasks:
            controller = Controller(flow=self)
            controller.run()


def get_flow() -> Optional[Flow]:
    """
    Loads the flow from the context. If no flow is found, returns None.
    """
    flow: Union[Flow, None] = ctx.get("flow")
    return flow


def get_flow_messages(limit: int = None) -> list[MessageType]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    if flow:
        return flow.thread.get_messages(limit=limit)
    else:
        return []
