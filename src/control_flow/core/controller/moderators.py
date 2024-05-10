import itertools
from typing import TYPE_CHECKING, Any, Generator

import marvin
from pydantic import BaseModel, Field

from control_flow.core.agent import Agent
from control_flow.core.flow import Flow, get_flow_messages
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


class AgentModerator(BaseModerator):
    agent: Agent
    participate: bool = Field(
        False,
        description="If True, the moderator can participate in the conversation. Default is False.",
    )

    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent=agent, **kwargs)

    def run(self, agents: list[Agent], tasks: list[Task]) -> Generator[Any, Any, Agent]:
        while True:
            history = get_flow_messages()

            with Flow():
                task = Task(
                    "Choose the next agent that should speak.",
                    instructions="""
                        You are acting as a moderator. Choose the next agent to
                        speak. Complete the task and stay silent. Do not post
                        any messages, even to confirm marking the task as
                        successful.
                        """,
                    result_type=[a.name for a in agents],
                    context=dict(agents=agents, history=history, tasks=tasks),
                    agents=[self.agent],
                    parent=None,
                )
                agent_name = task.run_until_complete()
                yield next(a for a in agents if a.name == agent_name)


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
                account any tasks, history, instructions, and tools.
                """,
                model_kwargs=dict(model=self.model) if self.model else None,
            )

            yield agent
