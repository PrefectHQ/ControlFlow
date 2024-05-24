import datetime
import json
import uuid
from typing import Any, List, Literal, Optional, Union

import litellm
from pydantic import (
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from controlflow.utilities.types import _OpenAIBaseType

# -----------------------------------------------
# Messages
# -----------------------------------------------


Role = Literal["system", "user", "assistant", "tool"]


class TextContent(_OpenAIBaseType):
    type: Literal["text"] = "text"
    text: str


class ImageDetails(_OpenAIBaseType):
    url: str
    detail: Literal["auto", "high", "low"] = "auto"


class ImageContent(_OpenAIBaseType):
    type: Literal["image_url"] = "image_url"
    image_url: ImageDetails


class ControlFlowMessage(_OpenAIBaseType):
    # ---- begin openai fields
    role: Role = Field(openai_field=True)
    _openai_fields: set[str] = {"role"}
    # ---- end openai fields

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, repr=False)
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
    llm_response: Optional[litellm.ModelResponse] = Field(None, repr=False)

    @field_validator("role", mode="before")
    def _lowercase_role(cls, v):
        if isinstance(v, str):
            v = v.lower()
        return v

    @field_validator("timestamp", mode="before")
    def _validate_timestamp(cls, v):
        if isinstance(v, int):
            v = datetime.datetime.fromtimestamp(v)
        return v

    @model_validator(mode="after")
    def _finalize(self):
        self._openai_fields = (
            getattr(super(), "_openai_fields", set()) | self._openai_fields
        )
        return self

    @field_serializer("timestamp")
    def _serialize_timestamp(self, timestamp: datetime.datetime):
        return timestamp.isoformat()

    def as_openai_message(self, include: set[str] = None, **kwargs) -> dict:
        include = self._openai_fields | (include or set())
        return self.model_dump(include=include, **kwargs)


class SystemMessage(ControlFlowMessage):
    # ---- begin openai fields
    role: Literal["system"] = "system"
    content: str
    name: Optional[str] = None
    _openai_fields = {"role", "content", "name"}


class UserMessage(ControlFlowMessage):
    # ---- begin openai fields
    role: Literal["user"] = "user"
    content: List[Union[TextContent, ImageContent]]
    name: Optional[str] = None
    _openai_fields = {"role", "content", "name"}
    # ---- end openai fields

    @field_validator("content", mode="before")
    def _validate_content(cls, v):
        if isinstance(v, str):
            v = [TextContent(text=v)]
        return v


class AssistantMessage(ControlFlowMessage):
    """A message from the assistant."""

    # ---- begin openai fields
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None
    _openai_fields = {"role", "content", "tool_calls"}
    # ---- end openai fields

    is_delta: bool = Field(
        default=False,
        description="If True, this message is a streamed delta, or chunk, of a full message.",
    )

    def has_tool_calls(self):
        return bool(self.tool_calls)


class ToolMessage(ControlFlowMessage):
    """A message for reporting the result of a tool call."""

    # ---- begin openai fields
    role: Literal["tool"] = "tool"
    content: str = Field(description="The string result of the tool call.")
    tool_call_id: str = Field(description="The ID of the tool call.")
    _openai_fields = {"role", "content", "tool_call_id"}
    # ---- end openai fields

    tool_call: "ToolCall" = Field(repr=False)
    tool_result: Any = Field(None, exclude=True)
    tool_metadata: dict = Field(default_factory=dict)


MessageType = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class ToolCall(_OpenAIBaseType):
    id: Optional[str]
    type: Literal["function"] = "function"
    function: "ToolCallFunction"


class ToolCallFunction(_OpenAIBaseType):
    name: Optional[str]
    arguments: str

    def json_arguments(self):
        return json.loads(self.arguments)


def as_cf_messages(
    messages: list[Union[litellm.Message, litellm.ModelResponse]],
) -> list[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]]:
    message_ta = TypeAdapter(
        Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]
    )

    result = []
    for msg in messages:
        if isinstance(msg, ControlFlowMessage):
            result.append(msg)
        elif isinstance(msg, litellm.Message):
            new_msg = message_ta.validate_python(msg.model_dump())
            result.append(new_msg)
        elif isinstance(msg, litellm.ModelResponse):
            for i, choice in enumerate(msg.choices):
                # handle delta messages streaming from the assistant
                if hasattr(choice, "delta"):
                    if choice.delta.role is None:
                        new_msg = AssistantMessage(is_delta=True)
                    else:
                        new_msg = AssistantMessage(
                            **choice.delta.model_dump(), is_delta=True
                        )
                else:
                    new_msg = message_ta.validate_python(choice.message.model_dump())
                new_msg.id = f"{msg.id}-{i}"
                new_msg.timestamp = msg.created
                new_msg.llm_response = msg
                result.append(new_msg)
        else:
            raise ValueError(f"Invalid message type: {type(msg)}")
    return result


def as_oai_messages(messages: list[Union[dict, ControlFlowMessage, litellm.Message]]):
    result = []
    for msg in messages:
        if isinstance(msg, ControlFlowMessage):
            result.append(msg.as_openai_message())
        elif isinstance(msg, (dict, litellm.Message)):
            result.append(msg)
        else:
            raise ValueError(f"Invalid message type: {type(msg)}")
    return result
