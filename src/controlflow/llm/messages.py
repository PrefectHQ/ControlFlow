import datetime
import re
import uuid
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import langchain_core.messages
from langchain_core.messages import InvalidToolCall, ToolCall
from pydantic.v1 import BaseModel as v1_BaseModel
from pydantic.v1 import Field as v1_Field
from pydantic.v1 import validator as v1_validator

from controlflow.utilities.jinja import jinja_env

if TYPE_CHECKING:
    from controlflow.agents.agent import Agent


class MessageMixin(langchain_core.messages.BaseMessage):
    class Config:
        validate_assignment = True

    # add timestamp
    timestamp: datetime.datetime = v1_Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def __init__(self, **data):
        # for some reason the id is not set if we add a default_factory
        if data.get("id") is None:
            data["id"] = uuid.uuid4().hex
        super().__init__(**data)

    @property
    def str_content(self) -> str:
        return str(self.content or "")

    def render(self, **kwargs) -> "MessageType":
        """
        Renders the content as a jinja template with the given keyword arguments
        and returns a new Message.
        """
        content = jinja_env.from_string(self.content).render(**kwargs)
        return self.copy(update=dict(content=content))


class AgentReference(v1_BaseModel):
    """
    A simplified representation of an agent for use in messages.

    ControlFlow agents that are passed to AgentReference fields should be automatically
    serialized to this format.
    """

    id: str = v1_Field(None)
    name: str = v1_Field(None)


class AIMessageMixin(MessageMixin):
    """
    Base class for AI messages and chunks.
    """

    role: Literal["ai"] = v1_Field("ai", exclude=True)
    # Agents are Pydantic v2 models, so we store them as dicts here.
    # they will be automatically converted.
    agent: Optional[AgentReference] = v1_Field(None, exclude=True)
    name: Optional[str] = v1_Field(None)

    def __init__(self, agent: "Agent" = None, **data):
        if agent is not None and data.get("name") is None:
            data["name"] = agent.name
        super().__init__(agent=agent, **data)

    @v1_validator("name", always=True)
    def _sanitize_name(cls, v):
        # sanitize name for API compatibility - OpenAI API only allows alphanumeric characters, dashes, and underscores
        if v is not None:
            v = re.sub(r"[^a-zA-Z0-9_-]", "-", v).strip("-")
        return v


class HumanMessage(langchain_core.messages.HumanMessage, MessageMixin):
    role: Literal["human"] = v1_Field("human", exclude=True)


class AIMessage(langchain_core.messages.AIMessage, AIMessageMixin):
    def __init__(self, **data):
        super().__init__(**data)

        # GPT-4 models somtimes use a hallucinated parallel tool calling mechanism
        # whose name is not compatible with the API's restrictions on tool names
        for tool_call in self.tool_calls:
            if tool_call["name"] == "multi_tool_use.parallel":
                tool_call["name"] = "multi_tool_use_parallel"

    def has_tool_calls(self) -> bool:
        return any(self.tool_calls)

    @classmethod
    def from_message(cls, message: langchain_core.messages.AIMessage, **kwargs):
        return cls(**dict(message) | kwargs | {"role": "ai"})


class AIMessageChunk(langchain_core.messages.AIMessageChunk, AIMessageMixin):
    def has_tool_calls(self) -> bool:
        return any(self.tool_call_chunks)

    @classmethod
    def from_chunk(
        cls, chunk: langchain_core.messages.AIMessageChunk, **kwargs
    ) -> "AIMessageChunk":
        return cls(**chunk.dict(exclude={"type"}) | kwargs | {"role": "ai"})

    def to_message(self, **kwargs) -> AIMessage:
        return AIMessage(**self.dict(exclude={"type"}) | kwargs)

    def __add__(self, other: Any) -> "AIMessageChunk":  # type: ignore
        result = super().__add__(other)
        result.timestamp = self.timestamp
        result.name = self.name
        result.agent = self.agent
        return result


class SystemMessage(langchain_core.messages.SystemMessage, MessageMixin):
    role: Literal["system"] = v1_Field("system", exclude=True)


class ToolMessage(langchain_core.messages.ToolMessage, MessageMixin):
    class Config:
        arbitrary_types_allowed = True

    role: Literal["tool"] = v1_Field("tool", exclude=True)

    tool_call: ToolCall = None
    tool_result: Any = v1_Field(exclude=True)
    tool_metadata: dict[str, Any] = v1_Field(default_factory=dict)
    agent: Optional[AgentReference] = v1_Field(None)


class InvalidToolMessage(ToolMessage):
    tool_call: InvalidToolCall
    agent: Optional[AgentReference] = v1_Field(None)


MessageType = Union[
    HumanMessage, AIMessage, SystemMessage, ToolMessage, InvalidToolMessage
]
