import marvin

from controlflow.core.agent import Agent
from controlflow.core.flow import get_flow_messages
from controlflow.core.task import Task
from controlflow.instructions import get_instructions


def round_robin(
    agents: list[Agent],
    tasks: list[Task],
    context: dict = None,
    iteration: int = 0,
) -> Agent:
    return agents[iteration % len(agents)]


def marvin_moderator(
    agents: list[Agent],
    tasks: list[Task],
    context: dict = None,
    iteration: int = 0,
    model: str = None,
) -> Agent:
    history = get_flow_messages()
    instructions = get_instructions()
    context = context or {}
    context.update(tasks=tasks, history=history, instructions=instructions)
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
