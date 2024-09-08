from typing import Any, Union

from prefect.context import TaskRunContext

from controlflow.agents.agent import Agent
from controlflow.flows import Flow, get_flow
from controlflow.orchestration.orchestrator import Orchestrator, TurnStrategy
from controlflow.tasks.task import Task
from controlflow.utilities.prefect import prefect_task


def get_task_run_name() -> str:
    context = TaskRunContext.get()
    tasks = context.parameters["tasks"]
    task_names = " | ".join(t.friendly_name() for t in tasks)
    return f"Run task{'s' if len(tasks) > 1 else ''}: {task_names}"


@prefect_task(task_run_name=get_task_run_name)
def run_tasks(
    tasks: list[Task],
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    raise_on_error: bool = True,
    max_calls_per_turn: int = None,
    max_turns: int = None,
) -> list[Any]:
    """
    Run a list of tasks.

    Returns a list of task results corresponding to the input tasks, or raises an error if any tasks failed.
    """
    flow = flow or get_flow() or Flow()

    orchestrator = Orchestrator(
        tasks=tasks,
        flow=flow,
        agent=agent,
        turn_strategy=turn_strategy,
    )
    orchestrator.run(
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
    )

    if raise_on_error and any(t.is_failed() for t in tasks):
        errors = [f"- {t.friendly_name()}: {t.result}" for t in tasks if t.is_failed()]
        if errors:
            raise ValueError(
                f"{len(errors)} task{'s' if len(errors) != 1 else ''} failed: "
                + "\n".join(errors)
            )


@prefect_task(task_run_name=get_task_run_name)
async def run_tasks_async(
    tasks: list[Task],
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    raise_on_error: bool = True,
    max_calls_per_turn: int = None,
    max_turns: int = None,
):
    """
    Run a list of tasks.
    """
    flow = flow or get_flow() or Flow()
    orchestrator = Orchestrator(
        tasks=tasks,
        flow=flow,
        agent=agent,
        turn_strategy=turn_strategy,
    )
    await orchestrator.run_async(
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
    )

    if raise_on_error and any(t.is_failed() for t in tasks):
        errors = [f"- {t.friendly_name()}: {t.result}" for t in tasks if t.is_failed()]
        if errors:
            raise ValueError(
                f"{len(errors)} task{'s' if len(errors) != 1 else ''} failed: "
                + "\n".join(errors)
            )


def _prep_tasks(
    objective_or_tasks: str | list[Task],
    **task_kwargs,
) -> tuple[list[Task], bool]:
    single_task = False

    if isinstance(objective_or_tasks, str):
        tasks = [Task(objective=objective_or_tasks, **task_kwargs)]
        single_task = True
    elif isinstance(objective_or_tasks, list):
        if task_kwargs:
            raise ValueError(
                "When providing a list of Tasks, do not pass any additional keyword arguments."
            )
        tasks = objective_or_tasks
    else:
        raise ValueError(
            f"Unrecognized type for `objective_or_tasks`: {type(objective_or_tasks)}. Expected a str, Task, or list[Task]."
        )

    return tasks, single_task


def run(
    objective: str | list[Task],
    *,
    turn_strategy: TurnStrategy = None,
    max_calls_per_turn: int = None,
    max_turns: int = None,
    raise_on_error: bool = True,
    **task_kwargs,
) -> Union[Any, list[Any]]:
    tasks, single_task = _prep_tasks(objective, **task_kwargs)

    run_tasks(
        tasks=tasks,
        raise_on_error=raise_on_error,
        turn_strategy=turn_strategy,
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
    )
    if single_task:
        return tasks[0].result
    else:
        return [t.result for t in tasks]


async def run_async(
    objective: str | list[Task],
    *,
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    max_calls_per_turn: int = None,
    max_turns: int = None,
    raise_on_error: bool = True,
    **task_kwargs,
) -> Union[Any, list[Any]]:
    tasks, single_task = _prep_tasks(objective, **task_kwargs)

    await run_tasks_async(
        tasks=tasks,
        flow=flow,
        agent=agent,
        turn_strategy=turn_strategy,
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
        raise_on_error=raise_on_error,
    )
    if single_task:
        return tasks[0].result
    else:
        return [t.result for t in tasks]
