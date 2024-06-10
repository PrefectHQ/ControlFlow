import datetime
import uuid
from typing import Any, Literal, Union

import langchain_core.messages
from langchain_core.messages import InvalidToolCall, ToolCall
from pydantic.v1 import Field as v1_Field

from controlflow.utilities.jinja import jinja_env


class MessageMixin(langchain_core.messages.BaseMessage):
    class Config:
        validate_assignment = True

    # default ID value
    id: str = v1_Field(default_factory=lambda: uuid.uuid4().hex)

    # add timestamp
    timestamp: datetime.datetime = v1_Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )

    def render(self, **kwargs) -> "MessageType":
        """
        Renders the content as a jinja template with the given keyword arguments
        and returns a new Message.
        """
        content = jinja_env.from_string(self.content).render(**kwargs)
        return self.copy(update=dict(content=content))


class HumanMessage(langchain_core.messages.HumanMessage, MessageMixin):
    role: Literal["human"] = v1_Field("human", exclude=True)


class AIMessage(langchain_core.messages.AIMessage, MessageMixin):
    role: Literal["ai"] = v1_Field("ai", exclude=True)

    def has_tool_calls(self) -> bool:
        return any(self.tool_calls)

    @classmethod
    def from_message(cls, message: langchain_core.messages.AIMessage, **kwargs):
        return cls(**dict(message) | kwargs | {"role": "ai"})


class AIMessageChunk(langchain_core.messages.AIMessageChunk, AIMessage):
    role: Literal["ai"] = v1_Field("ai", exclude=True)

    def has_tool_calls(self) -> bool:
        return any(self.tool_call_chunks)

    @classmethod
    def from_chunk(
        cls, chunk: langchain_core.messages.AIMessageChunk, **kwargs
    ) -> "AIMessageChunk":
        return cls(**dict(chunk) | kwargs | {"role": "ai"})

    def to_message(self, **kwargs) -> AIMessage:
        return AIMessage(**self.dict(exclude={"type"}) | kwargs)

    def __add__(self, other: Any) -> "AIMessageChunk":  # type: ignore
        result = super().__add__(other)
        result.timestamp = self.timestamp
        result.name = self.name
        return result


class SystemMessage(langchain_core.messages.SystemMessage, MessageMixin):
    role: Literal["system"] = v1_Field("system", exclude=True)


class ToolMessage(langchain_core.messages.ToolMessage, MessageMixin):
    class Config:
        arbitrary_types_allowed = True

    role: Literal["tool"] = v1_Field("tool", exclude=True)

    tool_call: ToolCall
    tool_result: Any = v1_Field(exclude=True)
    tool_metadata: dict[str, Any] = v1_Field(default_factory=dict)


class InvalidToolMessage(ToolMessage):
    tool_call: InvalidToolCall


MessageType = Union[
    HumanMessage, AIMessage, SystemMessage, ToolMessage, InvalidToolMessage
]
