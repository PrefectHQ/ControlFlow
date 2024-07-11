import datetime
import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.agents.agent import BaseAgent
    from controlflow.events.message_compiler import CompileContext
    from controlflow.llm.messages import BaseMessage
    from controlflow.tasks.task import Task

# This is a global variable that will be shared between all instances of InMemoryStore
IN_MEMORY_STORE = {}


class Event(ControlFlowModel):
    model_config: dict = dict(extra="allow")

    event: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: Optional[str] = None
    task_ids: set[str] = Field(default_factory=set)
    agent_ids: set[str] = Field(default_factory=set)
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    persist: bool = True

    def to_messages(self, context: "CompileContext") -> list["BaseMessage"]:
        return []

    def add_tasks(self, tasks: list["Task"]):
        for task in tasks:
            self.task_ids.add(task.id)

    def add_agents(self, agents: list["BaseAgent"]):
        from controlflow.agents.teams import Team

        for agent in agents:
            if agent.id in self.agent_ids:
                continue
            self.agent_ids.add(agent.id)
            if isinstance(agent, Team):
                self.add_agents(agent.agents)


class UnpersistedEvent(Event):
    model_config = dict(arbitrary_types_allowed=True)
    persist: bool = False
