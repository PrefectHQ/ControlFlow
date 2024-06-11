import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import Field, field_serializer

import controlflow
from controlflow.llm.models import get_default_model
from controlflow.tools.talk_to_human import talk_to_human
from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.core.task import Task
logger = logging.getLogger(__name__)


def get_default_agent() -> "Agent":
    return controlflow.default_agent


class Agent(ControlFlowModel):
    model_config = dict(arbitrary_types_allowed=True)
    name: str = Field(
        ...,
        description="The name of the agent. This is used to identify the agent in the system and should be unique per assigned task.",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    description: Optional[str] = Field(
        None, description="A description of the agent, visible to other agents."
    )
    instructions: Optional[str] = Field(
        None, description="Instructions for the agent, private to this agent."
    )
    tools: list[Callable] = Field(
        [], description="List of tools availble to the agent."
    )
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )

    # note: `model` should be typed as a BaseChatModel but V2 models can't have
    # V1 attributes without erroring, so we have to use Any.
    model: Any = Field(
        description="The LangChain BaseChatModel used by the agent. If not provided, the default model will be used.",
        default_factory=get_default_model,
        exclude=True,
    )

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        tools = controlflow.llm.tools.as_tools(tools)
        # tools are Pydantic 1 objects
        return [t.dict(include={"name", "description"}) for t in tools]

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def get_tools(self) -> list[Callable]:
        tools = self.tools.copy()
        if self.user_access:
            tools.append(talk_to_human)
        return tools

    def run(self, task: "Task"):
        return task.run_once(agent=self)

    async def run_async(self, task: "Task"):
        return await task.run_once_async(agent=self)


DEFAULT_AGENT = Agent(
    name="Marvin",
    instructions="""
        You are a diligent AI assistant. You complete 
        your tasks efficiently and without error.
        """,
)
