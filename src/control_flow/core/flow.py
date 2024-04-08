import functools
from typing import Callable

from marvin.beta.assistants import Thread
from marvin.utilities.logging import get_logger
from openai.types.beta.threads import Message
from prefect import flow as prefect_flow
from prefect import task as prefect_task
from pydantic import Field, field_validator

from control_flow.utilities.context import ctx
from control_flow.utilities.marvin import patch_marvin
from control_flow.utilities.types import AssistantTool, ControlFlowModel

logger = get_logger(__name__)


class Flow(ControlFlowModel):
    thread: Thread = Field(None, validate_default=True)
    tools: list[AssistantTool | Callable] = Field(
        [], description="Tools that will be available to every agent in the flow"
    )
    instructions: str | None = None
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

    def add_message(self, message: str):
        prefect_task(self.thread.add)(message)


def ai_flow(
    fn=None,
    *,
    thread: Thread = None,
    tools: list[AssistantTool | Callable] = None,
    instructions: str = None,
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
            instructions=instructions,
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
                "instructions": instructions,
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


def get_flow() -> Flow:
    """
    Loads the flow from the context.

    Will error if no flow is found in the context.
    """
    flow: Flow | None = ctx.get("flow")
    if not flow:
        raise ValueError("No flow found in context")
    return flow


def get_flow_messages(limit: int = None) -> list[Message]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    return flow.thread.get_messages(limit=limit)
