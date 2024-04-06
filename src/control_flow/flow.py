import functools
from typing import Callable, Optional, Union

from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.utilities.logging import get_logger
from prefect import flow as prefect_flow
from prefect import task as prefect_task
from pydantic import BaseModel, Field, field_validator

from control_flow.context import ctx

logger = get_logger(__name__)


class AIFlow(BaseModel):
    thread: Thread = Field(None, validate_default=True)
    assistant: Optional[Assistant] = Field(None, validate_default=True)
    tools: list[Union[AssistantTool, Callable]] = Field(None, validate_default=True)
    instructions: Optional[str] = None

    model_config: dict = dict(validate_assignment=True, extra="forbid")

    @field_validator("assistant", mode="before")
    def _load_assistant_from_ctx(cls, v):
        if v is None:
            v = ctx.get("assistant", None)
        return v

    @field_validator("thread", mode="before")
    def _load_thread_from_ctx(cls, v):
        if v is None:
            v = ctx.get("thread", None)
            if v is None:
                v = Thread()
        if not v.id:
            v.create()

        return v

    @field_validator("tools", mode="before")
    def _default_tools(cls, v):
        if v is None:
            v = []
        return v

    def add_message(self, message: str):
        prefect_task(self.thread.add)(message)


def ai_flow(
    fn=None,
    *,
    assistant: Assistant = None,
    thread: Thread = None,
    tools: list[Union[AssistantTool, Callable]] = None,
    instructions: str = None,
):
    """
    Prepare a function to be executed as a Control Flow flow.
    """

    if fn is None:
        return functools.partial(
            ai_flow,
            assistant=assistant,
            thread=thread,
            tools=tools,
            instructions=instructions,
        )

    @functools.wraps(fn)
    def wrapper(
        *args,
        _assistant: Assistant = None,
        _thread: Thread = None,
        _tools: list[Union[AssistantTool, Callable]] = None,
        _instructions: str = None,
        **kwargs,
    ):
        p_fn = prefect_flow(fn)
        flow_assistant = _assistant or assistant
        flow_thread = (
            _thread
            or thread
            or (flow_assistant.default_thread if flow_assistant else None)
            or Thread()
        )
        flow_instructions = _instructions or instructions
        flow_tools = _tools or tools
        flow_obj = AIFlow(
            thread=flow_thread,
            assistant=flow_assistant,
            tools=flow_tools,
            instructions=flow_instructions,
        )

        logger.info(
            f'Executing AI flow "{fn.__name__}" on thread "{flow_obj.thread.id}"'
        )

        with ctx(flow=flow_obj):
            return p_fn(*args, **kwargs)

    return wrapper
