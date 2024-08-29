from contextlib import ExitStack
from typing import Any, Optional

from pydantic import field_validator

from controlflow.agents.agent import Agent, AgentActions
from controlflow.events.base import Event
from controlflow.events.message_compiler import MessageCompiler
from controlflow.flows import Flow
from controlflow.llm.messages import BaseMessage
from controlflow.orchestration.handler import Handler
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, as_tools
from controlflow.utilities.general import ControlFlowModel

__all__ = [
    "AgentContext",
]


class AgentContext(ControlFlowModel):
    """
    The full context for an invocation of a Agent
    """

    model_config = dict(arbitrary_types_allowed=True)
    flow: Flow
    tasks: list[Task]
    tools: list[Any] = []
    instructions: list[str] = []
    prompts: list[str] = []
    handlers: list[Handler] = []
    _context: Optional[ExitStack] = None

    @field_validator("tools", mode="before")
    def _validate_tools(cls, v):
        if v:
            v = as_tools(v)
        return v

    def handle_event(self, event: Event, persist: bool = None):
        if persist is None:
            persist = event.persist
        for handler in self.handlers:
            handler.handle(event)
        if persist:
            self.flow.add_events([event])

    def add_handlers(self, handlers: list[Handler]):
        self.handlers = self.handlers + handlers

    def add_tools(self, tools: list[Tool]):
        self.tools = self.tools + tools

    def add_prompts(self, prompts: list[str]):
        self.prompts = self.prompts + prompts

    def add_instructions(self, instructions: list[str]):
        self.instructions = self.instructions + instructions

    def get_events(self, limit: Optional[int] = None) -> list[Event]:
        events = self.flow.get_events(limit=limit or 100)
        return events

    def compile_prompt(self, agent: Agent) -> str:
        from controlflow.orchestration.prompt_templates import (
            InstructionsTemplate,
            TasksTemplate,
            ToolTemplate,
        )

        prompts = [
            agent.get_prompt(context=self),
            self.flow.get_prompt(context=self),
            TasksTemplate(tasks=self.tasks, context=self).render(),
            ToolTemplate(tools=self.tools, context=self).render(),
            InstructionsTemplate(instructions=self.instructions, context=self).render(),
            *self.prompts,
        ]
        return "\n\n".join([p for p in prompts if p])

    def compile_messages(self, agent: Agent) -> list[BaseMessage]:
        events = self.get_events()
        compiler = MessageCompiler(
            events=events,
            llm_rules=agent.get_llm_rules(),
            system_prompt=self.compile_prompt(agent=agent),
        )
        messages = compiler.compile_to_messages(agent=agent)
        return messages


AgentActions.model_rebuild()
