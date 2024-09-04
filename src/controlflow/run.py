from controlflow.flows import Flow, get_flow
from controlflow.orchestration.orchestrator import Orchestrator, TurnStrategy
from controlflow.tasks.task import Task


def run(
    objective: str,
    *,
    turn_strategy: TurnStrategy = None,
    max_calls_per_turn: int = None,
    max_turns: int = None,
    **task_kwargs,
):
    task = Task(
        objective=objective,
        **task_kwargs,
    )
    return task.run(
        turn_strategy=turn_strategy,
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
    )


async def run_async(
    objective: str,
    *,
    turn_strategy: TurnStrategy = None,
    max_calls_per_turn: int = None,
    max_turns: int = None,
    **task_kwargs,
):
    task = Task(
        objective=objective,
        **task_kwargs,
    )
    return await task.run_async(
        turn_strategy=turn_strategy,
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
    )


def run_tasks(
    tasks: list[Task],
    flow: Flow = None,
    turn_strategy: TurnStrategy = None,
    **run_kwargs,
):
    """
    Convenience function to run a list of tasks to completion.
    """
    flow = flow or get_flow() or Flow()
    orchestrator = Orchestrator(tasks=tasks, flow=flow, turn_strategy=turn_strategy)
    return orchestrator.run(**run_kwargs)


async def run_tasks_async(
    tasks: list[Task],
    flow: Flow = None,
    turn_strategy: TurnStrategy = None,
    **run_kwargs,
):
    """
    Convenience function to run a list of tasks to completion asynchronously.
    """
    flow = flow or get_flow() or Flow()
    orchestrator = Orchestrator(tasks=tasks, flow=flow, turn_strategy=turn_strategy)
    return await orchestrator.run_async(**run_kwargs)
