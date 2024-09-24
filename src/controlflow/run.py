from typing import Any

from prefect.context import TaskRunContext

from controlflow.agents.agent import Agent
from controlflow.flows import Flow, get_flow
from controlflow.orchestration.handler import Handler
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
    raise_on_failure: bool = True,
    max_llm_calls: int = None,
    max_agent_turns: int = None,
    handlers: list[Handler] = None,
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
        handlers=handlers,
    )
    orchestrator.run(
        max_llm_calls=max_llm_calls,
        max_agent_turns=max_agent_turns,
    )

    if raise_on_failure and any(t.is_failed() for t in tasks):
        errors = [f"- {t.friendly_name()}: {t.result}" for t in tasks if t.is_failed()]
        if errors:
            raise ValueError(
                f"{len(errors)} task{'s' if len(errors) != 1 else ''} failed: "
                + "\n".join(errors)
            )

    return [t.result for t in tasks]


@prefect_task(task_run_name=get_task_run_name)
async def run_tasks_async(
    tasks: list[Task],
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    raise_on_failure: bool = True,
    max_llm_calls: int = None,
    max_agent_turns: int = None,
    handlers: list[Handler] = None,
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
        handlers=handlers,
    )
    await orchestrator.run_async(
        max_llm_calls=max_llm_calls,
        max_agent_turns=max_agent_turns,
    )

    if raise_on_failure and any(t.is_failed() for t in tasks):
        errors = [f"- {t.friendly_name()}: {t.result}" for t in tasks if t.is_failed()]
        if errors:
            raise ValueError(
                f"{len(errors)} task{'s' if len(errors) != 1 else ''} failed: "
                + "\n".join(errors)
            )

    return [t.result for t in tasks]


def run(
    objective: str,
    *,
    turn_strategy: TurnStrategy = None,
    max_llm_calls: int = None,
    max_agent_turns: int = None,
    raise_on_failure: bool = True,
    handlers: list[Handler] = None,
    **task_kwargs,
) -> Any:
    task = Task(objective=objective, **task_kwargs)
    results = run_tasks(
        tasks=[task],
        raise_on_failure=raise_on_failure,
        turn_strategy=turn_strategy,
        max_llm_calls=max_llm_calls,
        max_agent_turns=max_agent_turns,
        handlers=handlers,
    )
    return results[0]


async def run_async(
    objective: str,
    *,
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    max_llm_calls: int = None,
    max_agent_turns: int = None,
    raise_on_failure: bool = True,
    handlers: list[Handler] = None,
    **task_kwargs,
) -> Any:
    task = Task(objective=objective, **task_kwargs)
    results = await run_tasks_async(
        tasks=[task],
        flow=flow,
        agent=agent,
        turn_strategy=turn_strategy,
        max_llm_calls=max_llm_calls,
        max_agent_turns=max_agent_turns,
        raise_on_failure=raise_on_failure,
        handlers=handlers,
    )
    return results[0]
