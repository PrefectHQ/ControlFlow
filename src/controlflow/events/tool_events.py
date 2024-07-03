from typing import Literal

from pydantic import field_validator, model_validator

from controlflow.agents.agent import Agent
from controlflow.events.events import Event
from controlflow.events.message_compiler import EventContext
from controlflow.llm.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from controlflow.tools.tools import ToolCall, ToolResult
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


class ToolCallEvent(Event):
    event: Literal["tool-call"] = "tool-call"
    agent: Agent
    tool_call: ToolCall
    message: dict

    @field_validator("message", mode="before")
    def _message(cls, v):
        if isinstance(v, AIMessage):
            v = v.dict()
        v["type"] = "ai"
        return v

    @model_validator(mode="after")
    def _finalize(self):
        self.message["name"] = self.agent.name

    @property
    def ai_message(self) -> AIMessage:
        return AIMessage(**self.message)


class ToolResultEvent(Event):
    event: Literal["tool-result"] = "tool-result"
    agent: Agent
    tool_call: ToolCall
    tool_result: ToolResult

    def to_messages(self, context: EventContext) -> list[BaseMessage]:
        if self.agent.name == context.agent.name:
            return [
                ToolMessage(
                    content=self.tool_result.str_result,
                    tool_call_id=self.tool_call["id"],
                    name=self.agent.name,
                )
            ]
        elif not self.tool_result.is_private:
            return [
                SystemMessage(
                    content=f'The following {"failed" if self.tool_result.is_error else "successful"} '
                    f'tool result was received by "{self.agent.name}" with ID {self.agent.id}:'
                ),
                SystemMessage(content=self.tool_result.str_result),
            ]

        else:
            return []
