import logging
from typing import Callable, Optional, Union

from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from pydantic import Field

import controlflow
from controlflow.core.flow import Flow, get_flow
from controlflow.core.task import Task
from controlflow.tools.talk_to_human import talk_to_human
from controlflow.utilities.prefect import (
    wrap_prefect_tool,
)
from controlflow.utilities.types import Assistant, ControlFlowModel, ToolType

logger = logging.getLogger(__name__)


def get_default_agent() -> "Agent":
    return controlflow.default_agent


class AgentOLD(Assistant, ControlFlowModel, ExposeSyncMethodsMixin):
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


class Agent(ControlFlowModel, ExposeSyncMethodsMixin):
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
    model: str = Field(
        description="The model used by the agent. If not provided, the default model will be used.",
        default_factory=lambda: controlflow.settings.model,
    )

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def get_tools(self) -> list[Callable]:
        tools = self.tools.copy()
        if self.user_access:
            tools.append(talk_to_human)
        return tools

    # def say(
    #     self, messages: Union[str, dict], thread_id: str = None, history: History = None
    # ) -> Response:

    #     if thread_id is None:
    #         thread_id = self.default_thread_id
    #     if history is None:
    #         history = get_default_history()
    #     if not isinstance(messages, list):
    #         raise ValueError("Messages must be provided as a list.")

    #     messages = [
    #         Message(role="user", content=m) if isinstance(m, str) else m
    #         for m in messages
    #     ]
    #     history_messages = history.load_messages(thread_id=thread_id, limit=50)

    #     response = completion(
    #         messages=history_messages + messages,
    #         model=self.model,
    #         tools=self.tools,
    #     )
    #     history.save_messages(
    #         thread_id=thread_id,
    #         messages=messages + history_messages + response.messages,
    #     )
    #     return response


DEFAULT_AGENT = Agent(
    name="Marvin",
    instructions="""
        You are a diligent AI assistant. You complete 
        your tasks efficiently and without error.
        """,
)
