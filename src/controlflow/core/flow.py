import datetime
import uuid
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional, Union

from pydantic import Field

import controlflow
import controlflow.llm
from controlflow.core.agent import Agent
from controlflow.core.task import Task
from controlflow.llm.history import History, get_default_history
from controlflow.llm.messages import MessageType
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import prefect_flow_context
from controlflow.utilities.types import ControlFlowModel

logger = get_logger(__name__)


class Flow(ControlFlowModel):
    name: Optional[str] = None
    description: Optional[str] = None
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    history: History = Field(
        default_factory=controlflow.llm.history.get_default_history
    )
    tools: list[Callable] = Field(
        default_factory=list,
        description="Tools that will be available to every agent in the flow",
    )
    agents: list[Agent] = Field(
        description="The default agents for the flow. These agents will be used "
        "for any task that does not specify agents.",
        default_factory=list,
    )
    context: dict[str, Any] = {}
    tasks: dict[str, Task] = {}
    _cm_stack: list[contextmanager] = []

    def __init__(self, *, copy_parent_history: bool = True, **kwargs):
        """
        By default, the flow will copy the history from the parent flow if one
        exists. Because each flow is a new thread, new messages will not be shared
        between the parent and child flow.
        """
        super().__init__(**kwargs)
        parent = get_flow()
        if parent and copy_parent_history:
            self.add_messages(parent.get_messages())

    def __enter__(self):
        # use stack so we can enter the context multiple times
        cm = self.create_context()
        self._cm_stack.append(cm)
        return cm.__enter__()

    def __exit__(self, *exc_info):
        # exit the context manager
        return self._cm_stack.pop().__exit__(*exc_info)

    def get_messages(
        self,
        limit: int = None,
        before: datetime.datetime = None,
        after: datetime.datetime = None,
    ) -> list[MessageType]:
        return self.history.load_messages(
            thread_id=self.thread_id, limit=limit, before=before, after=after
        )

    def add_messages(self, messages: list[MessageType]):
        self.history.save_messages(thread_id=self.thread_id, messages=messages)

    def add_task(self, task: Task):
        if self.tasks.get(task.id, task) is not task:
            raise ValueError(
                f"A different task with id '{task.id}' already exists in flow."
            )
        self.tasks[task.id] = task

    @contextmanager
    def create_context(self, create_prefect_flow_context: bool = True):
        if create_prefect_flow_context:
            prefect_ctx = prefect_flow_context(name=self.name)
        else:
            prefect_ctx = nullcontext()
        with ctx(flow=self), prefect_ctx:
            yield self

    async def run_async(self):
        """
        Runs the flow asynchronously.
        """
        if self.tasks:
            controller = controlflow.Controller(flow=self)
            await controller.run_async()

    def run(self):
        """
        Runs the flow.
        """
        if self.tasks:
            controller = controlflow.Controller(flow=self)
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
    if limit is None:
        limit = 50
    flow = get_flow()
    if flow:
        return get_default_history().load_messages(flow.thread_id, limit=limit)
    else:
        return []
