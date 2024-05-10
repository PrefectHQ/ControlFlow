import itertools
from typing import TYPE_CHECKING, Any, Generator

import marvin
from pydantic import BaseModel

from control_flow.core.agent import Agent
from control_flow.core.flow import get_flow_messages
from control_flow.core.task import Task
from control_flow.instructions import get_instructions

if TYPE_CHECKING:
    from control_flow.core.agent import Agent


def round_robin(agents: list[Agent], tasks: list[Task]) -> Generator[Any, Any, Agent]:
    """
    Given a list of potential agents, delegate the tasks in a round-robin fashion.
    """
    cycle = itertools.cycle(agents)
    while True:
        yield next(cycle)


class BaseModerator(BaseModel):
    def __call__(
        self, agents: list[Agent], tasks: list[Task]
    ) -> Generator[Any, Any, Agent]:
        yield from self.run(agents=agents, tasks=tasks)


class Moderator(BaseModerator):
    model: str = None

    def run(self, agents: list[Agent], tasks: list[Task]) -> Generator[Any, Any, Agent]:
        while True:
            instructions = get_instructions()
            history = get_flow_messages()

            context = dict(
                tasks=tasks, messages=history, global_instructions=instructions
            )
            agent = marvin.classify(
                context,
                agents,
                instructions="""
                Given the conversation context, choose the AI agent most
                qualified to take the next turn at completing the tasks. Take into
                account any tasks, instructions, and tools.
                """,
                model_kwargs=dict(model=self.model) if self.model else None,
            )

            yield agent
