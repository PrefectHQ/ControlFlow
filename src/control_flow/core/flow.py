from contextlib import contextmanager
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

    @contextmanager
    def _context(self):
        with ctx(flow=self, tasks=[]):
            yield self

    def __enter__(self):
        self.__cm = self._context()
        return self.__cm.__enter__()

    def __exit__(self, *exc_info):
        return self.__cm.__exit__(*exc_info)


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
