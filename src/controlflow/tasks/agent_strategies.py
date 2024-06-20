from controlflow.agents import Agent
from controlflow.flows import Flow, get_flow_messages
from controlflow.instructions import get_instructions
from controlflow.llm.classify import classify
from controlflow.tasks.task import Task


def round_robin(
    agents: list[Agent],
    task: Task,
    flow: Flow,
    **kwargs,
) -> Agent:
    return agents[task._iteration % len(agents)]


def moderator(
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

    agent = classify(
        context,
        labels=agents,
        instructions="""
            Given the context, choose the AI agent best suited to take the
            next turn at completing the tasks in the task graph. Take into account
            any descriptions, tasks, history, instructions, and tools. Focus on
            agents assigned to upstream dependencies or subtasks that need to be
            completed before their downstream/parents can be completed. An agent
            can only work on a task that it is assigned to.
            """,
        model=model,
    )
    return agent
