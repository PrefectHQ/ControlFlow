import functools
from typing import Callable, List, Optional, Union

from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.utilities.logging import get_logger
from prefect import flow as prefect_flow
from pydantic import BaseModel, Field, field_validator

from control_flow.context import ctx

from .task import AITask

logger = get_logger(__name__)


class AIFlow(BaseModel):
    tasks: List[AITask] = []
    thread: Thread = Field(None, validate_default=True)
    assistant: Assistant = Field(None, validate_default=True)
    tools: list[Union[AssistantTool, Callable]] = Field(None, validate_default=True)
    instructions: Optional[str] = None

    model_config: dict = dict(validate_assignment=True, extra="forbid")

    @field_validator("assistant", mode="before")
    def _load_assistant_from_ctx(cls, v):
        if v is None:
            v = ctx.get("assistant", None)
            if v is None:
                v = Assistant()
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

    def add_task(self, task: AITask):
        if task.id is None:
            task.id = len(self.tasks) + 1
        elif task.id in {t.id for t in self.tasks}:
            raise ValueError(f"Task with id {task.id} already exists.")
        self.tasks.append(task)

    def get_task_by_id(self, task_id: int) -> Optional[AITask]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def update_task(self, task_id: int, status: str, result: str = None):
        task = self.get_task_by_id(task_id)
        if task:
            task.update(status=status, result=result)

    def add_message(self, message: str):
        self.thread.add(message)


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
        if flow_assistant is None:
            flow_assistant = Assistant()
        flow_thread = _thread or thread or flow_assistant.default_thread
        flow_instructions = _instructions or instructions
        flow_tools = _tools or tools
        flow_obj = AIFlow(
            thread=flow_thread,
            assistant=flow_assistant,
            tools=flow_tools,
            instructions=flow_instructions,
        )
        with ctx(flow=flow_obj):
            return p_fn(*args, **kwargs)

    return wrapper
