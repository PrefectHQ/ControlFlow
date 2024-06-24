import datetime
import re
import uuid
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import langchain_core.messages
from langchain_core.messages.tool import InvalidToolCall, ToolCall, ToolCallChunk
from pydantic import Field, field_validator

from controlflow.utilities.jinja import jinja_env
from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.agents.agent import Agent


class BaseMessage(ControlFlowModel):
    """
    ControlFlow uses Message objects that are similar to LangChain messages, but more purpose built.

    Note that LangChain messages are Pydantic V1 models, while ControlFlow messages are Pydantic V2 models.
    """

    id: Optional[str] = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
    )
    role: str
    content: Union[str, list[Union[str, dict]]]
    name: Optional[str] = None

    # private attr to hold the original langchain message
    _langchain_message: Optional[
        Union[
            langchain_core.messages.BaseMessage,
            langchain_core.messages.BaseMessageChunk,
        ]
    ] = None

    @field_validator("name")
    def _sanitize_name(cls, v):
        # sanitize name for API compatibility - OpenAI API only allows alphanumeric characters, dashes, and underscores
        if v is not None:
            v = re.sub(r"[^a-zA-Z0-9_-]", "-", v).strip("-")
        return v

    def __init__(
        self,
        content: Union[str, list[Union[str, dict]]],
        _langchain_message: Optional[
            Union[
                langchain_core.messages.BaseMessage,
                langchain_core.messages.BaseMessageChunk,
            ]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """Pass in content as positional arg."""
        super().__init__(content=content, **kwargs)
        self._langchain_message = _langchain_message

    @property
    def str_content(self) -> str:
        if not isinstance(self.content, str):
            return str(self.content)
        return self.content

    def render(self, **kwargs) -> "MessageType":
        """
        Renders the content as a jinja template with the given keyword arguments
        and returns a new Message.
        """
        content = jinja_env.from_string(self.content).render(**kwargs)
        return self.model_copy(update=dict(content=content))

    @classmethod
    def _langchain_message_kwargs(
        cls, message: langchain_core.messages.BaseMessage
    ) -> "BaseMessage":
        return message.dict(include={"content", "id", "name"}) | dict(
            _langchain_message=message
        )

    @classmethod
    def from_langchain_message(message: langchain_core.messages.BaseMessage, **kwargs):
        raise NotImplementedError()

    def to_langchain_message(self) -> langchain_core.messages.BaseMessage:
        raise NotImplementedError()


class AgentReference(ControlFlowModel):
    name: str


class AgentMessageMixin(ControlFlowModel):
    agent: Optional[AgentReference] = None

    @field_validator("agent", mode="before")
    def _validate_agent(cls, v):
        from controlflow.agents.agent import Agent

        if isinstance(v, Agent):
            return AgentReference(name=v.name)
        return v

    def __init__(self, *args, agent: "Agent" = None, **data):
        if agent is not None and data.get("name") is None:
            data["name"] = agent.name
        super().__init__(*args, agent=agent, **data)


class AIMessage(BaseMessage, AgentMessageMixin):
    role: Literal["ai"] = "ai"
    tool_calls: list[ToolCall] = []

    is_delta: bool = False

    def __init__(self, *args, **data):
        super().__init__(*args, **data)

        # GPT-4 models somtimes use a hallucinated parallel tool calling mechanism
        # whose name is not compatible with the API's restrictions on tool names
        for tool_call in self.tool_calls:
            if tool_call["name"] == "multi_tool_use.parallel":
                tool_call["name"] = "multi_tool_use_parallel"

    def has_tool_calls(self) -> bool:
        return any(self.tool_calls)

    @classmethod
    def from_langchain_message(
        cls,
        message: langchain_core.messages.AIMessage,
        **kwargs,
    ):
        data = dict(
            **cls._langchain_message_kwargs(message),
            tool_calls=message.tool_calls + getattr(message, "invalid_tool_calls", []),
        )

        return cls(**data | kwargs)

    def to_langchain_message(
        self,
    ) -> langchain_core.messages.AIMessage:
        if self._langchain_message is not None:
            return self._langchain_message
        return langchain_core.messages.AIMessage(
            content=self.content, tool_calls=self.tool_calls, id=self.id, name=self.name
        )


class AIMessageChunk(AIMessage):
    tool_calls: list[ToolCallChunk] = []

    @classmethod
    def from_langchain_message(
        cls,
        message: Union[
            langchain_core.messages.AIMessageChunk, langchain_core.messages.AIMessage
        ],
        **kwargs,
    ):
        if isinstance(message, langchain_core.messages.AIMessageChunk):
            tool_calls = message.tool_call_chunks
        elif isinstance(message, langchain_core.messages.AIMessage):
            tool_calls = []
            for i, call in enumerate(message.tool_calls):
                tool_calls.append(
                    ToolCallChunk(
                        id=call["id"],
                        name=call["name"],
                        args=str(call["args"]) if call["args"] else None,
                        index=call.get("index", i),
                    )
                )
        data = dict(
            **cls._langchain_message_kwargs(message),
            tool_calls=tool_calls,
            is_delta=True,
        )

        return cls(**data | kwargs)

    def to_langchain_message(
        self,
    ) -> langchain_core.messages.AIMessageChunk:
        if self._langchain_message is not None:
            return self._langchain_message
        return langchain_core.messages.AIMessageChunk(
            content=self.content,
            tool_call_chunks=self.tool_calls,
            id=self.id,
            name=self.name,
        )


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"

    @classmethod
    def from_langchain_message(
        cls, message: langchain_core.messages.HumanMessage, **kwargs
    ):
        return cls(**cls._langchain_message_kwargs(message) | kwargs)

    def to_langchain_message(self) -> langchain_core.messages.BaseMessage:
        if self._langchain_message is not None:
            return self._langchain_message
        return langchain_core.messages.HumanMessage(
            content=self.content, id=self.id, name=self.name
        )


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"

    @classmethod
    def from_langchain_message(
        cls, message: langchain_core.messages.SystemMessage, **kwargs
    ):
        return cls(**cls._langchain_message_kwargs(message) | kwargs)

    def to_langchain_message(self) -> langchain_core.messages.BaseMessage:
        if self._langchain_message is not None:
            return self._langchain_message
        return langchain_core.messages.SystemMessage(
            content=self.content, id=self.id, name=self.name
        )


class ToolMessage(BaseMessage, AgentMessageMixin):
    model_config = dict(arbitrary_types_allowed=True)
    role: Literal["tool"] = "tool"
    tool_call_id: str
    tool_call: Union[ToolCall, InvalidToolCall]
    tool_result: Any = Field(None, exclude=True)
    tool_metadata: dict[str, Any] = {}
    is_error: bool = False

    def to_langchain_message(self) -> langchain_core.messages.BaseMessage:
        if self._langchain_message is not None:
            return self._langchain_message
        else:
            return langchain_core.messages.ToolMessage(
                id=self.id,
                name=self.name,
                content=self.content,
                tool_call_id=self.tool_call_id,
            )


MessageType = Union[UserMessage, AIMessage, SystemMessage, ToolMessage]
