import abc
import logging
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Optional,
)

from langchain_core.language_models import BaseChatModel
from pydantic import Field, field_serializer

import controlflow
from controlflow.events.base import Event
from controlflow.instructions import get_instructions
from controlflow.llm.messages import AIMessage, BaseMessage
from controlflow.llm.rules import LLMRules
from controlflow.tools.tools import (
    as_lc_tools,
    as_tools,
    handle_tool_call,
    handle_tool_call_async,
)
from controlflow.utilities.context import ctx
from controlflow.utilities.general import ControlFlowModel, hash_objects

from .memory import Memory

if TYPE_CHECKING:
    from controlflow.tools.tools import Tool
logger = logging.getLogger(__name__)


class Agent(ControlFlowModel, abc.ABC):
    """
    Class for objects that can be used as agents in a flow
    """

    model_config = dict(arbitrary_types_allowed=True)

    id: str = Field(None)
    name: str = Field(description="The name of the agent.")
    description: Optional[str] = Field(
        None, description="A description of the agent, visible to other agents."
    )
    instructions: Optional[str] = Field(
        "You are a diligent AI assistant. You complete your tasks efficiently and without error.",
        description="Instructions for the agent, private to this agent.",
    )
    prompt: Optional[str] = Field(
        None,
        description="A system template for the agent. The template should be formatted as a jinja2 template.",
    )
    tools: list[Callable] = Field(
        [], description="List of tools available to the agent."
    )
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    memory: Optional[Memory] = Field(
        default=None,
        # default_factory=ThreadMemory,
        description="The memory object used by the agent. If not specified, an in-memory memory object will be used. Pass None to disable memory.",
        exclude=True,
    )

    # note: `model` should be typed as Optional[BaseChatModel] but V2 models can't have
    # V1 attributes without erroring, so we have to use Any.
    model: Optional[Any] = Field(
        None,
        description="The LangChain BaseChatModel used by the agent. If not provided, the default model will be used.",
        exclude=True,
    )

    _cm_stack: list[contextmanager] = []

    def __init__(self, name: str = None, **kwargs):
        if name is not None:
            kwargs["name"] = name

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions")
                or "" + "\n" + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)

        if not self.id:
            self.id = self._generate_id()

    def __hash__(self) -> int:
        return id(self)

    def _generate_id(self):
        """
        Helper function to generate a stable, short, semi-unique ID for the agent.
        """
        return hash_objects(
            (
                type(self).__name__,
                self.name,
                self.description,
                self.prompt,
                self.instructions,
            )
        )

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Callable]):
        tools = controlflow.tools.as_tools(tools)
        # tools are Pydantic 1 objects
        return [t.dict(include={"name", "description"}) for t in tools]

    def serialize_for_prompt(self) -> dict:
        dct = self.model_dump(
            include={"name", "id", "description", "tools", "user_access"}
        )
        if not dct["user_access"]:
            dct.pop("user_access")
        return dct

    def get_model(self, tools: list["Tool"] = None) -> BaseChatModel:
        """
        Retrieve the LLM model for this agent
        """
        model = self.model or controlflow.defaults.model
        if model is None:
            raise ValueError(
                f"Agent {self.name}: No model provided and no default model could be loaded."
            )
        if tools:
            model = model.bind_tools(as_lc_tools(tools))
        return model

    def get_llm_rules(self) -> LLMRules:
        """
        Retrieve the LLM rules for this agent's model
        """
        return controlflow.llm.rules.rules_for_model(self.get_model())

    def get_tools(self) -> list["Tool"]:
        from controlflow.tools.talk_to_user import talk_to_user

        tools = self.tools.copy()
        if self.user_access:
            tools.append(talk_to_user)
        if self.memory is not None:
            tools.extend(self.memory.get_tools())

        return as_tools(tools)

    def get_prompt(self) -> str:
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.AgentTemplate(template=self.prompt, agent=self)
        return template.render()

    @contextmanager
    def create_context(self):
        with ctx(agent=self):
            yield self

    def __enter__(self):
        self._cm_stack.append(self.create_context())
        return self._cm_stack[-1].__enter__()

    def __exit__(self, *exc_info):
        return self._cm_stack.pop().__exit__(*exc_info)

    def run(self, *args, agents: list["Agent"] = None, **kwargs):
        agents = agents or [] + [self]
        task = controlflow.Task(*args, agents=agents, **kwargs)
        return task.run()

    async def run_async(self, *args, agents: list["Agent"] = None, **kwargs):
        agents = agents or [] + [self]
        task = controlflow.Task(*args, agents=agents, **kwargs)
        return await task.run_async()

    def _run_model(
        self,
        messages: list[BaseMessage],
        tools: list["Tool"],
        stream: bool = True,
    ) -> Generator[Event, None, None]:
        from controlflow.events.events import (
            AgentMessage,
            AgentMessageDelta,
            ToolCallEvent,
            ToolResultEvent,
        )

        model = self.get_model(tools=tools)

        if stream:
            response = None
            for delta in model.stream(messages):
                if response is None:
                    response = delta
                else:
                    response += delta

                yield AgentMessageDelta(agent=self, delta=delta, snapshot=response)

        else:
            response: AIMessage = model.invoke(messages)

        yield AgentMessage(agent=self, message=response)

        for tool_call in response.tool_calls + response.invalid_tool_calls:
            yield ToolCallEvent(agent=self, tool_call=tool_call)
            result = handle_tool_call(tool_call, tools=tools)
            yield ToolResultEvent(agent=self, tool_call=tool_call, tool_result=result)

    async def _run_model_async(
        self,
        messages: list[BaseMessage],
        tools: list["Tool"],
        stream: bool = True,
    ) -> AsyncGenerator[Event, None]:
        from controlflow.events.events import (
            AgentMessage,
            AgentMessageDelta,
            ToolCallEvent,
            ToolResultEvent,
        )

        model = self.get_model(tools=tools)

        if stream:
            response = None
            async for delta in model.astream(messages):
                if response is None:
                    response = delta
                else:
                    response += delta

                yield AgentMessageDelta(agent=self, delta=delta, snapshot=response)

        else:
            response: AIMessage = await model.ainvoke(messages)

        yield AgentMessage(agent=self, message=response)

        for tool_call in response.tool_calls + response.invalid_tool_calls:
            yield ToolCallEvent(agent=self, tool_call=tool_call, message=response)
            result = await handle_tool_call_async(tool_call, tools=tools)
            yield ToolResultEvent(agent=self, tool_call=tool_call, tool_result=result)
