from typing import TYPE_CHECKING, Literal, Optional, Union

import pydantic_core
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
from controlflow.tools.tools import InvalidToolCall, Tool, ToolCall
from controlflow.tools.tools import ToolResult as ToolResultPayload
from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.events.message_compiler import CompileContext
logger = get_logger(__name__)

ORCHESTRATOR_PREFIX = "The following message is from the orchestrator."


class OrchestratorMessage(Event):
    """
    Messages from the orchestrator to agents.
    """

    event: Literal["orchestrator-message"] = "orchestrator-message"
    content: Union[str, list[Union[str, dict]]]
    prefix: Optional[str] = ORCHESTRATOR_PREFIX
    name: Optional[str] = None

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        messages = []
        # if self.prefix:
        #     messages.append(SystemMessage(content=self.prefix))
        messages.append(
            HumanMessage(content=f"({self.prefix})\n\n{self.content}", name=self.name)
        )
        return messages


class UserMessage(Event):
    event: Literal["user-message"] = "user-message"
    content: Union[str, list[Union[str, dict]]]

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        return [HumanMessage(content=self.content)]


class AgentMessage(Event):
    event: Literal["agent-message"] = "agent-message"
    agent: Agent
    message: dict

    @field_validator("message", mode="before")
    def _as_message_dict(cls, v):
        if isinstance(v, BaseMessage):
            v = v.model_dump()
        v["type"] = "ai"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        self.message["name"] = self.agent.name
        return self

    @property
    def ai_message(self) -> AIMessage:
        return AIMessage(**self.message)

    def to_tool_calls(self, tools: list[Tool]) -> list["AgentToolCall"]:
        calls = []
        for tool_call in (
            self.message["tool_calls"] + self.message["invalid_tool_calls"]
        ):
            tool = next((t for t in tools if t.name == tool_call.get("name")), None)
            if tool:
                calls.append(
                    AgentToolCall(
                        agent=self.agent,
                        tool_call=tool_call,
                        tool=tool,
                        args=tool_call["args"],
                        agent_message_id=self.message.get("id"),
                    )
                )
        return calls

    def to_content(self) -> Optional["AgentContent"]:
        if self.message.get("content"):
            return AgentContent(
                agent=self.agent,
                content=self.message["content"],
                agent_message_id=self.message.get("id"),
            )

    def all_related_events(self, tools: list[Tool]) -> list[Event]:
        content = self.to_content()
        return [self] + ([content] if content else []) + self.to_tool_calls(tools)

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
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
    event: Literal["agent-message-delta"] = "agent-message-delta"

    agent: Agent
    message_delta: dict
    message_snapshot: dict

    @field_validator("message_delta", "message_snapshot", mode="before")
    def _as_message_dict(cls, v):
        if isinstance(v, BaseMessage):
            v = v.model_dump()
        v["type"] = "AIMessageChunk"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        self.message_delta["name"] = self.agent.name
        self.message_snapshot["name"] = self.agent.name
        return self

    def to_tool_call_deltas(self, tools: list[Tool]) -> list["AgentToolCallDelta"]:
        deltas = []
        for call_delta in self.message_delta.get("tool_call_chunks", []):
            # First match chunks by index because streaming chunks come in sequence (0,1,2...)
            # and this index lets us correlate deltas to their snapshots during streaming
            chunk_snapshot = next(
                (
                    c
                    for c in self.message_snapshot.get("tool_call_chunks", [])
                    if c.get("index", -1) == call_delta.get("index", -2)
                ),
                None,
            )

            if chunk_snapshot and chunk_snapshot.get("id"):
                # Once we have the matching chunk, use its ID to find the full tool call
                # The full tool calls contain properly parsed arguments (as Python dicts)
                # while chunks just contain raw JSON strings
                call_snapshot = next(
                    (
                        c
                        for c in self.message_snapshot["tool_calls"]
                        if c.get("id") == chunk_snapshot["id"]
                    ),
                    None,
                )

                if call_snapshot:
                    tool = next(
                        (t for t in tools if t.name == call_snapshot.get("name")), None
                    )
                    # Use call_snapshot.args which is already parsed into a Python dict
                    # This avoids issues with pydantic's more limited JSON parser
                    deltas.append(
                        AgentToolCallDelta(
                            agent=self.agent,
                            tool_call_delta=call_delta,
                            tool_call_snapshot=call_snapshot,
                            tool=tool,
                            args=call_snapshot.get("args", {}),
                            agent_message_id=self.message_snapshot.get("id"),
                        )
                    )
        return deltas

    def to_content_delta(self) -> Optional["AgentContentDelta"]:
        if self.message_delta.get("content"):
            return AgentContentDelta(
                agent=self.agent,
                content_delta=self.message_delta["content"],
                content_snapshot=self.message_snapshot["content"],
                agent_message_id=self.message_snapshot.get("id"),
            )

    def all_related_events(self, tools: list[Tool]) -> list[Event]:
        content_delta = self.to_content_delta()
        return (
            [self]
            + ([content_delta] if content_delta else [])
            + self.to_tool_call_deltas(tools)
        )


class AgentContent(UnpersistedEvent):
    event: Literal["agent-content"] = "agent-content"
    agent: Agent
    agent_message_id: Optional[str] = None
    content: Union[str, list[Union[str, dict]]]


class AgentContentDelta(UnpersistedEvent):
    event: Literal["agent-content-delta"] = "agent-content-delta"
    agent: Agent
    agent_message_id: Optional[str] = None
    content_delta: Union[str, list[Union[str, dict]]]
    content_snapshot: Union[str, list[Union[str, dict]]]


class AgentToolCall(Event):
    event: Literal["agent-tool-call"] = "agent-tool-call"
    agent: Agent
    agent_message_id: Optional[str] = None
    tool_call: Union[ToolCall, InvalidToolCall]
    tool: Optional[Tool] = None
    args: dict = {}


class AgentToolCallDelta(UnpersistedEvent):
    event: Literal["agent-tool-call-delta"] = "agent-tool-call-delta"
    agent: Agent
    agent_message_id: Optional[str] = None
    tool_call_delta: dict
    tool_call_snapshot: dict
    tool: Optional[Tool] = None
    args: dict = {}


class EndTurn(Event):
    event: Literal["end-turn"] = "end-turn"
    agent: Agent
    next_agent_name: Optional[str] = None


class ToolResult(Event):
    event: Literal["tool-result"] = "tool-result"
    agent: Agent
    tool_result: ToolResultPayload

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        if self.agent.name == context.agent.name:
            return [
                ToolMessage(
                    content=self.tool_result.str_result,
                    tool_call_id=self.tool_result.tool_call["id"],
                    name=self.agent.name,
                )
            ]
        else:
            return OrchestratorMessage(
                prefix=f'Agent "{self.agent.name}" with ID {self.agent.id} made a tool '
                f'call: {self.tool_result.tool_call}. The tool{" failed and" if self.tool_result.is_error else " "} '
                f'produced this result:',
                content=self.tool_result.str_result,
                name=self.agent.name,
            ).to_messages(context)
