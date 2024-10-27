from typing import TYPE_CHECKING, Literal, Optional, Union, List



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
    Represents messages from the orchestrator to agents.
    """

    event: Literal["orchestrator-message"] = "orchestrator-message"
    content: Union[str, List[Union[str, dict]]]
    prefix: Optional[str] = ORCHESTRATOR_PREFIX
    name: Optional[str] = None

    def to_messages(self, context: "CompileContext") -> List[BaseMessage]:
        logger.debug("Creating orchestrator messages with prefix: %s", self.prefix)
        messages = [
            HumanMessage(content=f"({self.prefix})\n\n{self.content}", name=self.name)
        ]
        return messages


class UserMessage(Event):
    """
    Represents messages from the user.
    """

    event: Literal["user-message"] = "user-message"
    content: Union[str, List[Union[str, dict]]]

    def to_messages(self, context: "CompileContext") -> List[BaseMessage]:
        logger.debug("Creating user message: %s", self.content)
        return [HumanMessage(content=self.content)]


class AgentMessage(Event):
    """
    Represents messages from an agent.
    """

    event: Literal["agent-message"] = "agent-message"
    agent: Agent
    message: dict

    @field_validator("message", mode="before")
    def validate_message(cls, v):
        if isinstance(v, BaseMessage):
            v = v.model_dump()
        v["type"] = "ai"
        return v

    @model_validator(mode="after")
    def finalize_message(self):
        self.message["name"] = self.agent.name
        logger.debug("Finalized agent message for agent: %s", self.agent.name)
        return self

    @property
    def ai_message(self) -> AIMessage:
        return AIMessage(**self.message)

    def to_messages(self, context: "CompileContext") -> List[BaseMessage]:
        if self.agent.name == context.agent.name:
            return [self.ai_message]
        
        if self.message.get("content"):
            return OrchestratorMessage(
                prefix=f'The following message was posted by Agent "{self.agent.name}" with ID {self.agent.id}',
                content=self.message["content"],
                name=self.agent.name,
            ).to_messages(context)
        
        return []


class AgentMessageDelta(UnpersistedEvent):
    """
    Represents a delta change in an agent's message.
    """

    event: Literal["agent-message-delta"] = "agent-message-delta"
    agent: Agent
    delta: dict
    snapshot: dict

    @field_validator("delta", "snapshot", mode="before")
    def validate_delta_and_snapshot(cls, v):
        if isinstance(v, BaseMessage):
            v = v.model_dump()
        v["type"] = "AIMessageChunk"
        return v

    @model_validator(mode="after")
    def finalize_delta_and_snapshot(self):
        self.delta["name"] = self.agent.name
        self.snapshot["name"] = self.agent.name
        logger.debug("Finalized delta and snapshot for agent: %s", self.agent.name)
        return self

    @property
    def delta_message(self) -> AIMessageChunk:
        return AIMessageChunk(**self.delta)

    @property
    def snapshot_message(self) -> AIMessage:
        return AIMessage(**{**self.snapshot, "type": "ai"})


class EndTurn(Event):
    """
    Represents the end of an agent's turn.
    """

    event: Literal["end-turn"] = "end-turn"
    agent: Agent
    next_agent_name: Optional[str] = None


class ToolCallEvent(Event):
    """
    Represents a tool call made by an agent.
    """

    event: Literal["tool-call"] = "tool-call"
    agent: Agent
    tool_call: Union[ToolCall, InvalidToolCall]


class ToolResultEvent(Event):
    """
    Represents the result of a tool call made by an agent.
    """

    event: Literal["tool-result"] = "tool-result"
    agent: Agent
    tool_call: Union[ToolCall, InvalidToolCall]
    tool_result: ToolResult

    def to_messages(self, context: "CompileContext") -> List[BaseMessage]:
        if self.agent.name == context.agent.name:
            logger.debug("Creating tool result message for agent: %s", self.agent.name)
            return [
                ToolMessage(
                    content=self.tool_result.str_result,
                    tool_call_id=self.tool_call["id"],
                    name=self.agent.name,
                )
            ]
        
        logger.debug("Creating orchestrator message for tool result from agent: %s", self.agent.name)
        return OrchestratorMessage(
            prefix=f'Agent "{self.agent.name}" with ID {self.agent.id} made a tool call: {self.tool_call}. '
                   f'The tool{" failed and" if self.tool_result.is_error else " "} produced this result:',
            content=self.tool_result.str_result,
            name=self.agent.name,
        ).to_messages(context)
