import logging
import random
import re
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import Field, field_serializer

import controlflow
from controlflow.llm.models import get_default_model
from controlflow.tools.talk_to_human import talk_to_human
from controlflow.utilities.context import ctx
from controlflow.utilities.types import ControlFlowModel

from .memory import Memory
from .names import NAMES

if TYPE_CHECKING:
    from controlflow.core.task import Task
logger = logging.getLogger(__name__)


def get_default_agent() -> "Agent":
    return controlflow.default_agent


def sanitize_name(name):
    """
    Replace any invalid characters with `-`, due to restrictions on names in the API
    """
    sanitized_string = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    return sanitized_string


class Agent(ControlFlowModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:5]))
    model_config = dict(arbitrary_types_allowed=True)
    name: str = Field(
        description="The name of the agent.",
        default_factory=lambda: random.choice(NAMES),
    )
    description: Optional[str] = Field(
        None, description="A description of the agent, visible to other agents."
    )
    instructions: Optional[str] = Field(
        "You are a diligent AI assistant. You complete your tasks efficiently and without error.",
        description="Instructions for the agent, private to this agent.",
    )
    tools: list[Callable] = Field(
        [], description="List of tools availble to the agent."
    )
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    memory: Optional[Memory] = Field(
        default=None,
        # default_factory=ThreadMemory,
        description="The memory object used by the agent. If not specified, an in-memory memory object will be used. Pass None to disable memory.",
    )

    # note: `model` should be typed as a BaseChatModel but V2 models can't have
    # V1 attributes without erroring, so we have to use Any.
    model: Any = Field(
        description="The LangChain BaseChatModel used by the agent. If not provided, the default model will be used.",
        default_factory=get_default_model,
        exclude=True,
    )

    _cm_stack: list[contextmanager] = []

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        tools = controlflow.llm.tools.as_tools(tools)
        # tools are Pydantic 1 objects
        return [t.dict(include={"name", "description"}) for t in tools]

    @field_serializer("name")
    def _serialize_name(self, name: str):
        return sanitize_name(name)

    def __init__(self, name=None, **kwargs):
        if name is not None:
            kwargs["name"] = name
        super().__init__(**kwargs)

    def get_tools(self) -> list[Callable]:
        tools = self.tools.copy()
        if self.user_access:
            tools.append(talk_to_human)
        if self.memory is not None:
            tools.extend(self.memory.get_tools())
        return tools

    @contextmanager
    def create_context(self):
        with ctx(agent=self):
            yield self

    def __enter__(self):
        # use stack so we can enter the context multiple times
        self._cm_stack.append(self.create_context())
        return self._cm_stack[-1].__enter__()

    def __exit__(self, *exc_info):
        return self._cm_stack.pop().__exit__(*exc_info)

    def run(self, task: "Task"):
        return task.run_once(agent=self)

    async def run_async(self, task: "Task"):
        return await task.run_once_async(agent=self)


DEFAULT_AGENT = Agent(name="Marvin")
