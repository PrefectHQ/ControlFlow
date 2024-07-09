from contextlib import ExitStack
from functools import partial, wraps
from typing import Callable, Optional

from pydantic import Field

from controlflow.agents.agent import Agent, BaseAgent
from controlflow.events.base import Event
from controlflow.events.message_compiler import MessageCompiler
from controlflow.flows import Flow
from controlflow.llm.messages import BaseMessage
from controlflow.orchestration.handler import Handler
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool
from controlflow.utilities.context import ctx
from controlflow.utilities.types import ControlFlowModel

__all__ = [
    "AgentContext",
    "get_context",
    "provide_agent_context",
]


class Prompts(ControlFlowModel):
    agent: Optional[str] = None
    team: Optional[str] = None
    tasks: Optional[str] = None
    flow: Optional[str] = None
    instructions: list[str] = []


class AgentContext(ControlFlowModel):
    """
    The full context for an invocation of a BaseAgent
    """

    model_config = dict(arbitrary_types_allowed=True)
    agent: BaseAgent
    flow: Flow
    tasks: list[Task]
    tools: list[Tool] = []
    handlers: list[Handler] = []
    prompts: Prompts = Field(default_factory=Prompts)
    _context: Optional[ExitStack] = None

    def with_agent(self, agent: BaseAgent) -> "AgentContext":
        return self.model_copy(update={"agent": agent})

    def handle_event(self, event: Event, agent: Agent = None, persist: bool = None):
        if persist is None:
            persist = event.persist

        event.thread_id = self.flow.thread_id
        event.add_tasks(self.tasks)
        event.add_agents([agent] if agent else [])

        for handler in self.handlers:
            handler.handle(event)
        if persist:
            self.flow.add_events([event])

    def add_handlers(self, handlers: list[Handler]):
        self.handlers.extend(handlers)

    def add_tools(self, tools: list[Tool]):
        self.tools.extend(tools)

    def set_agent_prompt(self, prompt: str):
        self.prompts.agent = prompt

    def set_team_prompt(self, prompt: str):
        self.prompts.team = prompt

    def set_tasks_prompt(self, prompt: str):
        self.prompts.tasks = prompt

    def set_flow_prompt(self, prompt: str):
        self.prompts.flow = prompt

    def add_instructions(self, instructions: list[str]):
        self.prompts.instructions.extend(instructions)

    def get_events(self, agents: list[Agent] = None) -> list[Event]:
        upstream_tasks = [
            t for t in self.flow.graph.upstream_tasks(self.tasks) if not t.private
        ]
        events = self.flow.get_events(agents=agents, tasks=upstream_tasks)

        return events

    def compile_prompt(self) -> str:
        from controlflow.orchestration.prompts import InstructionsTemplate

        prompts = [
            self.prompts.agent,
            self.prompts.team,
            self.prompts.tasks,
            self.prompts.flow,
            InstructionsTemplate(instructions=self.prompts.instructions).render(),
        ]
        return "\n\n".join([p for p in prompts if p])

    def compile_messages(self, agent: Agent) -> list[BaseMessage]:
        events = self.get_events(agents=[agent])
        compiler = MessageCompiler(
            events=events,
            llm_rules=agent.get_llm_rules(),
            system_prompt=self.compile_prompt(),
        )
        messages = compiler.compile_to_messages(agent=agent)
        return messages

    def __enter__(self):
        self._context = ExitStack()
        self._context.enter_context(ctx(agent_context=self))
        return self

    def __exit__(self, *exc_info):
        self._context.close()
        return False


def get_context() -> Optional[AgentContext]:
    return ctx.get("agent_context")


def provide_agent_context(fn: Callable = None, *, context_kwarg: str = None):
    if fn is None:
        return partial(provide_agent_context, context_kwarg=context_kwarg)

    context_kwarg = context_kwarg or "context"

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if context_kwarg not in kwargs:
            if context := get_context():
                kwargs[context_kwarg] = context
        return fn(*args, **kwargs)

    return wrapper
