import datetime
import inspect
import json
import uuid
from enum import Enum
from functools import partial, update_wrapper
from typing import Any, Callable, List, Literal, Optional, Union

import litellm
import pydantic
from marvin.beta.assistants import Assistant, Thread
from marvin.beta.assistants.assistants import AssistantTool
from marvin.types import FunctionTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    TypeAdapter,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
    validator,
)
from sqlalchemy import desc
from traitlets import default

# flag for unset defaults
NOTSET = "__NOTSET__"


ToolType = Union[FunctionTool, AssistantTool, Callable]


class ControlFlowModel(BaseModel):
    model_config = dict(validate_assignment=True, extra="forbid")


class PandasDataFrame(ControlFlowModel):
    """Schema for a pandas dataframe"""

    data: Union[
        list[list[Union[str, int, float, bool]]],
        dict[str, list[Union[str, int, float, bool]]],
    ]
    columns: list[str] = None
    index: list[str] = None
    dtype: dict[str, str] = None


class PandasSeries(ControlFlowModel):
    """Schema for a pandas series"""

    data: list[Union[str, int, float]]
    index: list[str] = None
    name: str = None
    dtype: str = None


class ToolFunction(ControlFlowModel):
    name: str
    parameters: dict
    description: str = ""


class Tool(ControlFlowModel):
    type: Literal["function"] = "function"
    function: ToolFunction
    _fn: Callable = PrivateAttr()

    def __init__(self, *, _fn: Callable, **kwargs):
        super().__init__(**kwargs)
        self._fn = _fn

    @classmethod
    def from_function(
        cls, fn: Callable, name: Optional[str] = None, description: Optional[str] = None
    ):
        if name is None and fn.__name__ == "<lambda>":
            name = "__lambda__"

        return cls(
            function=ToolFunction(
                name=name or fn.__name__,
                description=inspect.cleandoc(description or fn.__doc__ or ""),
                parameters=pydantic.TypeAdapter(
                    fn, config=pydantic.ConfigDict(arbitrary_types_allowed=True)
                ).json_schema(),
            ),
            _fn=fn,
        )

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


# -----------------------------------------------
# Messages
# -----------------------------------------------


class _OpenAIBaseType(ControlFlowModel):
    model_config = dict(extra="allow")


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
    # ---- end openai fields


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

    tool_call: "ToolCall" = Field(cf_field=True, repr=False)
    tool_result: Any = Field(None, cf_field=True, exclude=True)
    tool_failed: bool = Field(False, cf_field=True)


MessageType = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class ToolCallFunction(_OpenAIBaseType):
    name: Optional[str]
    arguments: str

    def json_arguments(self):
        return json.loads(self.arguments)


class ToolCall(_OpenAIBaseType):
    id: Optional[str]
    type: Literal["function"] = "function"
    function: ToolCallFunction


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
