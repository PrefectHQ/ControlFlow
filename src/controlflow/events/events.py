from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import ConfigDict, field_validator, model_validator

from controlflow.agents.agent import Agent
from controlflow.events.base import Event, UnpersistedEvent
from controlflow.llm.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from controlflow.tools.tools import InvalidToolCall, ToolCall, ToolResult
from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.events.message_compiler import CompileContext
logger = get_logger(__name__)

ORCHESTRATOR_PREFIX = "The following message is from the orchestrator."


class OrchestratorMessage(Event):
    """
    Represents a message from the orchestrator to agents.
    
    Attributes:
        event: Literal identifier for orchestrator messages.
        content: The message content, either a string or a list of strings/dicts.
        prefix: An optional prefix to specify the source of the message.
        name: An optional name associated with the message sender.
    """

    event: Literal["orchestrator-message"] = "orchestrator-message"
    content: Union[str, list[Union[str, dict]]]
    prefix: Optional[str] = ORCHESTRATOR_PREFIX
    name: Optional[str] = None

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        """
        Converts the orchestrator message into a list of BaseMessage instances.

        Args:
            context: The context for message compilation.

        Returns:
            A list of BaseMessage objects representing the orchestrator's message.
        """
        messages = []
        messages.append(
            HumanMessage(content=f"({self.prefix})\n\n{self.content}", name=self.name)
        )
        return messages


class UserMessage(Event):
    """
    Represents a message sent by a user.
    
    Attributes:
        event: Literal identifier for user messages.
        content: The message content, either a string or a list of strings/dicts.
    """

    event: Literal["user-message"] = "user-message"
    content: Union[str, list[Union[str, dict]]]

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        """
        Converts the user message into a list of BaseMessage instances.

        Args:
            context: The context for message compilation.

        Returns:
            A list containing a single HumanMessage object.
        """
        return [HumanMessage(content=self.content)]


class AgentMessage(Event):
    """
    Represents a message sent by an agent.
    
    Attributes:
        event: Literal identifier for agent messages.
        agent: The agent sending the message.
        message: The message content, in dictionary format.
    """

    event: Literal["agent-message"] = "agent-message"
    agent: Agent
    message: dict

    @field_validator("message", mode="before")
    def _message(cls, v):
        """
        Validates and converts the message format, setting its type to "ai" if needed.

        Args:
            v: The initial message content.

        Returns:
            The validated message content.
        """
        if isinstance(v, BaseMessage):
            v = v.model_dump()
        v["type"] = "ai"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        """
        Finalizes the message by setting the agent's name.
        
        Returns:
            The updated message with agent's name added.
        """
        self.message["name"] = self.agent.name
        return self

    @property
    def ai_message(self) -> AIMessage:
        """
        Returns the message as an AIMessage object.

        Returns:
            An instance of AIMessage.
        """
        return AIMessage(**self.message)

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        """
        Converts the agent message into a list of BaseMessage instances based on the context.

        Args:
            context: The context for message compilation.

        Returns:
            A list of BaseMessage objects, depending on whether the agent matches the context agent.
        """
        if self.agent.name == context.agent.name:
            return [self.ai_message]
        elif self.message["content"]:
            return OrchestratorMessage(
                prefix=f'The following message was posted by Agent "{self.agent.name}" with ID {self.agent.id}',
                content=self.message["content"],
                name=self.agent.name,
            ).to_messages(context)
        else:
            return []


class AgentMessageDelta(UnpersistedEvent):
    """
    Represents an incremental update (delta) to an agent's message.
    
    Attributes:
        event: Literal identifier for agent message deltas.
        agent: The agent associated with the delta.
        delta: The delta content, in dictionary format.
        snapshot: The snapshot content representing the complete message at the current state.
    """

    event: Literal["agent-message-delta"] = "agent-message-delta"

    agent: Agent
    delta: dict
    snapshot: dict

    @field_validator("delta", "snapshot", mode="before")
    def _message(cls, v):
        """
        Validates and converts the delta and snapshot content format, setting type to "AIMessageChunk".

        Args:
            v: The initial delta or snapshot content.

        Returns:
            The validated content.
        """
        if isinstance(v, BaseMessage):
            v = v.model_dump()
        v["type"] = "AIMessageChunk"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        """
        Finalizes the delta and snapshot by setting the agent's name.

        Returns:
            The updated delta and snapshot with agent's name added.
        """
        self.delta["name"] = self.agent.name
        self.snapshot["name"] = self.agent.name
        return self

    @property
    def delta_message(self) -> AIMessageChunk:
        """
        Returns the delta as an AIMessageChunk object.

        Returns:
            An instance of AIMessageChunk.
        """
        return AIMessageChunk(**self.delta)

    @property
    def snapshot_message(self) -> AIMessage:
        """
        Returns the snapshot as an AIMessage object.

        Returns:
            An instance of AIMessage.
        """
        return AIMessage(**self.snapshot | {"type": "ai"})


class EndTurn(Event):
    """
    Represents an event signaling the end of an agent's turn.
    
    Attributes:
        event: Literal identifier for end-turn events.
        agent: The agent ending their turn.
        next_agent_name: Optional name of the next agent to act.
    """

    event: Literal["end-turn"] = "end-turn"
    agent: Agent
    next_agent_name: Optional[str] = None


class ToolCallEvent(Event):
    """
    Represents an event where an agent makes a tool call.
    
    Attributes:
        event: Literal identifier for tool call events.
        agent: The agent making the tool call.
        tool_call: The tool call, either a valid ToolCall or InvalidToolCall.
    """

    event: Literal["tool-call"] = "tool-call"
    agent: Agent
    tool_call: Union[ToolCall, InvalidToolCall]


class ToolResultEvent(Event):
    """
    Represents an event where a tool call produces a result.
    
    Attributes:
        event: Literal identifier for tool result events.
        agent: The agent receiving the tool result.
        tool_call: The initial tool call.
        tool_result: The result produced by the tool call.
    """

    event: Literal["tool-result"] = "tool-result"
    agent: Agent
    tool_call: Union[ToolCall, InvalidToolCall]
    tool_result: ToolResult

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        """
        Converts the tool result event into a list of BaseMessage instances.

        Args:
            context: The context for message compilation.

        Returns:
            A list of BaseMessage objects representing the tool result, tailored to the agent and context.
        """
        if self.agent.name == context.agent.name:
            return [
                ToolMessage(
                    content=self.tool_result.str_result,
                    tool_call_id=self.tool_call["id"],
                    name=self.agent.name,
                )
            ]
        else:
            return OrchestratorMessage(
                prefix=f'Agent "{self.agent.name}" with ID {self.agent.id} made a tool '
                f'call: {self.tool_call}. The tool{" failed and" if self.tool_result.is_error else " "} '
                f'produced this result:',
                content=self.tool_result.str_result,
                name=self.agent.name,
            ).to_messages(context)
