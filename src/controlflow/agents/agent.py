import logging
import random
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Generator, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import Field, field_serializer

import controlflow
from controlflow.events.events import Event
from controlflow.instructions import get_instructions
from controlflow.llm.messages import AIMessage, BaseMessage
from controlflow.llm.rules import LLMRules
from controlflow.tools.tools import handle_tool_call_async
from controlflow.utilities.context import ctx
from controlflow.utilities.types import ControlFlowModel

from .memory import Memory
from .names import NAMES

if TYPE_CHECKING:
    from controlflow.tasks.task import Task
    from controlflow.tools.tools import Tool

logger = logging.getLogger(__name__)


class Agent(ControlFlowModel):
    model_config = dict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:5]))
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

    # note: `model` should be typed as Optional[BaseChatModel] but V2 models can't have
    # V1 attributes without erroring, so we have to use Any.
    model: Optional[Any] = Field(
        None,
        description="The LangChain BaseChatModel used by the agent. If not provided, the default model will be used.",
        exclude=True,
    )

    _cm_stack: list[contextmanager] = []

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        tools = controlflow.tools.as_tools(tools)
        # tools are Pydantic 1 objects
        return [t.dict(include={"name", "description"}) for t in tools]

    def __init__(self, name=None, **kwargs):
        if name is not None:
            kwargs["name"] = name

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions")
                or "" + "\n" + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)

    def serialize_for_prompt(self) -> dict:
        dct = self.model_dump(
            include={"name", "id", "description", "tools", "user_access"}
        )
        # seeing user access = False can confuse agents on tasks with user access
        if not dct["user_access"]:
            dct.pop("user_access")
        return dct

    def get_model(self) -> BaseChatModel:
        """
        Retrieve the LLM model for this agent
        """
        model = self.model or controlflow.defaults.model
        if model is None:
            raise ValueError(
                f"Agent {self.name}: No model provided and no default model could be loaded."
            )
        return model

    def get_llm_rules(self) -> LLMRules:
        """
        Retrieve the LLM rules for this agent's model
        """
        return controlflow.llm.rules.rules_for_model(self.get_model())

    def get_tools(self) -> list[Callable]:
        from controlflow.tools.talk_to_user import talk_to_user

        tools = self.tools.copy()
        if self.user_access:
            tools.append(talk_to_user)
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

    def run_once(self, task: "Task"):
        return task.run_once(agents=[self])

    async def run_once_async(self, task: "Task"):
        return await task.run_once_async(agents=[self])

    def run(self, task: "Task"):
        return task.run(agents=[self])

    async def run_async(self, task: "Task"):
        return await task.run_async(agents=[self])

    def _run_model(
        self,
        messages: list[BaseMessage],
        additional_tools: list["Tool"] = None,
        stream: bool = True,
    ) -> Generator[Event, None, None]:
        from controlflow.events.agent_events import (
            AgentMessageDeltaEvent,
            AgentMessageEvent,
        )
        from controlflow.events.tool_events import ToolCallEvent, ToolResultEvent
        from controlflow.tools.tools import as_tools, handle_tool_call

        model = self.get_model()

        tools = as_tools(self.get_tools() + (additional_tools or []))
        if tools:
            model = model.bind_tools([t.to_lc_tool() for t in tools])

        if stream:
            response = None
            for delta in model.stream(messages):
                if response is None:
                    response = delta
                else:
                    response += delta
                yield AgentMessageDeltaEvent(agent=self, delta=delta, snapshot=response)

        else:
            response: AIMessage = model.invoke(messages)

        yield AgentMessageEvent(agent=self, message=response)

        for tool_call in response.tool_calls + response.invalid_tool_calls:
            yield ToolCallEvent(agent=self, tool_call=tool_call, message=response)

            result = handle_tool_call(tool_call, tools=tools)
            yield ToolResultEvent(agent=self, tool_call=tool_call, tool_result=result)

    async def _run_model_async(
        self,
        messages: list[BaseMessage],
        additional_tools: list["Tool"] = None,
        stream: bool = True,
    ) -> AsyncGenerator[Event, None]:
        from controlflow.events.agent_events import (
            AgentMessageDeltaEvent,
            AgentMessageEvent,
        )
        from controlflow.events.tool_events import ToolCallEvent, ToolResultEvent
        from controlflow.tools.tools import as_tools

        model = self.get_model()

        tools = as_tools(self.get_tools() + (additional_tools or []))
        if tools:
            model = model.bind_tools([t.to_lc_tool() for t in tools])

        if stream:
            response = None
            async for delta in model.astream(messages):
                if response is None:
                    response = delta
                else:
                    response += delta
                yield AgentMessageDeltaEvent(agent=self, delta=delta, snapshot=response)

        else:
            response: AIMessage = model.invoke(messages)

        yield AgentMessageEvent(agent=self, message=response)

        for tool_call in response.tool_calls + response.invalid_tool_calls:
            yield ToolCallEvent(agent=self, tool_call=tool_call, message=response)

            result = await handle_tool_call_async(tool_call, tools=tools)
            yield ToolResultEvent(agent=self, tool_call=tool_call, tool_result=result)


DEFAULT_AGENT = Agent(name="Marvin")
