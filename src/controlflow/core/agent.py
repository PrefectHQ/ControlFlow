import logging
from typing import Callable, Optional, Union

from litellm import Message
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from pydantic import Field

from controlflow.core.flow import Flow, get_flow
from controlflow.core.task import Task
from controlflow.llm.completions import Response, completion, completion_async
from controlflow.tools.talk_to_human import talk_to_human
from controlflow.utilities.prefect import (
    wrap_prefect_tool,
)
from controlflow.utilities.types import Assistant, ControlFlowModel, ToolType

logger = logging.getLogger(__name__)


def default_agent():
    return Agent(
        name="Marvin",
        instructions="""
            You are a diligent AI assistant. You complete 
            your tasks efficiently and without error.
            """,
    )


class Agent(Assistant, ControlFlowModel, ExposeSyncMethodsMixin):
    name: str
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )

    def get_tools(self) -> list[ToolType]:
        tools = super().get_tools()
        if self.user_access:
            tools.append(talk_to_human)

        return [wrap_prefect_tool(tool) for tool in tools]

    @expose_sync_method("run")
    async def run_async(
        self,
        tasks: Union[list[Task], Task, None] = None,
        flow: Flow = None,
    ):
        from controlflow.core.controller import Controller

        if isinstance(tasks, Task):
            tasks = [tasks]

        flow = flow or get_flow()

        if not flow:
            raise ValueError(
                "Agents must be run within a flow context or with a flow argument."
            )

        controller = Controller(agents=[self], tasks=tasks or [], flow=flow)
        return await controller.run_agent_async(agent=self)

    def __hash__(self):
        return id(self)


class LiteAgent(ControlFlowModel, ExposeSyncMethodsMixin):
    name: str = Field(
        ...,
        description="The name of the agent. This is used to identify the agent in the system and should be unique per assigned task.",
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
    model: Optional[str] = Field(
        None,
        description="The model used by the agent. If not provided, the default model will be used.",
    )

    async def say_async(self, messages: Union[str, dict]) -> Response:
        if not isinstance(messages, list):
            raise ValueError("Messages must be provided as a list.")

        messages = [
            Message(role="user", content=m) if isinstance(m, str) else m
            for m in messages
        ]

        return await completion_async(
            messages=messages, model=self.model, tools=self.tools
        )

    async def say(self, messages: Union[str, dict]) -> Response:
        if not isinstance(messages, list):
            raise ValueError("Messages must be provided as a list.")

        messages = [
            Message(role="user", content=m) if isinstance(m, str) else m
            for m in messages
        ]

        return completion(messages=messages, model=self.model, tools=self.tools)
