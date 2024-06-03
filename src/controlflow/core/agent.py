import logging
from typing import AsyncGenerator, Callable, Generator, Optional, Union

from pydantic import Field

import controlflow
from controlflow.core.task import Task
from controlflow.llm.completions import completion, completion_async
from controlflow.llm.handlers import CompletionEvent, CompletionHandler
from controlflow.llm.messages import ControlFlowMessage
from controlflow.tools.talk_to_human import talk_to_human
from controlflow.utilities.types import ControlFlowModel

logger = logging.getLogger(__name__)


def get_default_agent() -> "Agent":
    return controlflow.default_agent


class Agent(ControlFlowModel):
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
        default_factory=lambda: controlflow.settings.llm_model,
    )

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

    def completion(
        self,
        messages: list[ControlFlowMessage],
        tools: list[Callable] = None,
        handlers: list[CompletionHandler] = None,
        message_preprocessor: Optional[Callable] = None,
        stream: bool = False,
    ) -> Union[list[ControlFlowMessage], Generator[CompletionEvent, None, None]]:
        """
        Run the agent on the given messages.
        """

        if tools is None:
            tools = self.get_tools()

        response = completion(
            messages=messages,
            model=self.model,
            tools=tools,
            handlers=handlers,
            max_iterations=1,
            assistant_name=self.name,
            message_preprocessor=message_preprocessor,
            stream=stream,
        )

        return response

    async def completion_async(
        self,
        messages: list[ControlFlowMessage],
        tools: list[Callable] = None,
        handlers: list[CompletionHandler] = None,
        message_preprocessor: Optional[Callable] = None,
        stream: bool = False,
    ) -> Union[list[ControlFlowMessage], AsyncGenerator[CompletionEvent, None]]:
        """
        Run the agent on the given messages.
        """

        if tools is None:
            tools = self.get_tools()

        response = await completion_async(
            messages=messages,
            model=self.model,
            tools=tools,
            handlers=handlers,
            max_iterations=1,
            assistant_name=self.name,
            message_preprocessor=message_preprocessor,
            stream=stream,
        )

        return response


DEFAULT_AGENT = Agent(
    name="Marvin",
    instructions="""
        You are a diligent AI assistant. You complete 
        your tasks efficiently and without error.
        """,
)
