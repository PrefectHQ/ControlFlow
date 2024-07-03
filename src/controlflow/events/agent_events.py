from typing import Literal, Optional

from pydantic import field_validator, model_validator

from controlflow.agents.agent import Agent
from controlflow.events.events import Event, UnpersistedEvent
from controlflow.events.message_compiler import EventContext
from controlflow.llm.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


class SelectAgentEvent(Event):
    event: Literal["select-agent"] = "select-agent"
    agent: Agent

    # def to_messages(self, context: EventContext) -> list[BaseMessage]:
    #     return [
    #         SystemMessage(
    #             content=f'Agent "{self.agent.name}" with ID {self.agent.id} was selected.'
    #         )
    #     ]


class SystemMessageEvent(Event):
    event: Literal["system-message"] = "system-message"
    content: str

    def to_messages(self, context: EventContext) -> list[BaseMessage]:
        return [SystemMessage(content=self.content)]


class UserMessageEvent(Event):
    event: Literal["user-message"] = "user-message"
    content: str

    def to_messages(self, context: EventContext) -> list[BaseMessage]:
        return [HumanMessage(content=self.content)]


class AgentMessageEvent(Event):
    event: Literal["agent-message"] = "agent-message"
    agent: Agent
    message: dict

    @field_validator("message", mode="before")
    def _message(cls, v):
        if isinstance(v, BaseMessage):
            v = v.dict()
        v["type"] = "ai"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        self.message["name"] = self.agent.name

    @property
    def ai_message(self) -> AIMessage:
        return AIMessage(**self.message)

    def to_messages(self, context: EventContext) -> list[BaseMessage]:
        if self.agent.name == context.agent.name:
            return [self.ai_message]
        elif self.message["content"]:
            return [
                SystemMessage(
                    content=f'The following message was posted by Agent "{self.agent.name}" '
                    f"with ID {self.agent.id}:",
                ),
                HumanMessage(
                    # ensure this is stringified to avoid issues with inline tool calls
                    content=str(self.message["content"]),
                    name=self.agent.name,
                ),
            ]
        else:
            return []


class AgentMessageDeltaEvent(UnpersistedEvent):
    event: Literal["agent-message-delta"] = "agent-message-delta"

    agent: Agent
    delta: dict
    snapshot: dict

    @field_validator("delta", "snapshot", mode="before")
    def _message(cls, v):
        if isinstance(v, BaseMessage):
            v = v.dict()
        v["type"] = "AIMessageChunk"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        self.delta["name"] = self.agent.name
        self.snapshot["name"] = self.agent.name

    @property
    def delta_message(self) -> AIMessageChunk:
        return AIMessageChunk(**self.delta)

    @property
    def snapshot_message(self) -> AIMessage:
        return AIMessage(**self.snapshot | {"type": "ai"})


class EndTurnEvent(Event):
    event: Literal["end-turn"] = "end-turn"
    agent: Agent
    next_agent_name: Optional[str] = None
