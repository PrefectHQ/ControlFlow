import itertools
from typing import TYPE_CHECKING, Any, Generator

import marvin
from pydantic import BaseModel, Field

from control_flow.core.agent import Agent
from control_flow.core.flow import Flow, get_flow_messages
from control_flow.core.task import Task

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
                agent_name = task.run()
                yield next(a for a in agents if a.name == agent_name)


def marvin_moderator(
    agents: list[Agent],
    tasks: list[Task],
    context: dict = None,
    model: str = None,
) -> Agent:
    context = context or {}
    context.update(tasks=tasks)
    agent = marvin.classify(
        context,
        agents,
        instructions="""
            Given the context, choose the AI agent best suited to take the
            next turn at completing the tasks in the task graph. Take into account
            any descriptions, tasks, history, instructions, and tools. Focus on
            agents assigned to upstream dependencies or subtasks that need to be
            completed before their downstream/parents can be completed. An agent
            can only work on a task that it is assigned to.
            """,
        model_kwargs=dict(model=model) if model else None,
    )
    return agent
