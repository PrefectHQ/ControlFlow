from typing import Any, AsyncIterator, Callable, Iterator, Optional, Union

import controlflow
from controlflow.agents.agent import Agent
from controlflow.events.events import Event
from controlflow.flows import Flow, get_flow
from controlflow.orchestration.conditions import RunContext, RunEndCondition
from controlflow.orchestration.handler import AsyncHandler, Handler
from controlflow.orchestration.orchestrator import Orchestrator, TurnStrategy
from controlflow.stream import Stream, filter_events_async, filter_events_sync
from controlflow.tasks.task import Task
from controlflow.utilities.prefect import prefect_task


def run_tasks(
    tasks: list[Task],
    instructions: str = None,
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    raise_on_failure: bool = True,
    max_llm_calls: int = None,
    max_agent_turns: int = None,
    handlers: list[Handler] = None,
    model_kwargs: Optional[dict] = None,
    run_until: Optional[Union[RunEndCondition, Callable[[RunContext], bool]]] = None,
    stream: Union[bool, Stream] = False,
) -> Union[list[Any], Iterator[tuple[Event, Any, Optional[Any]]]]:
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

    with controlflow.instructions(instructions):
        result = orchestrator.run(
            max_llm_calls=max_llm_calls,
            max_agent_turns=max_agent_turns,
            model_kwargs=model_kwargs,
            run_until=run_until,
            stream=bool(stream),
        )

        if stream:
            # Convert True to ALL filter, otherwise use provided filter
            stream_filter = Stream.ALL if stream is True else stream
            return filter_events_sync(result, stream_filter)

    if raise_on_failure and any(t.is_failed() for t in tasks):
        errors = [f"- {t.friendly_name()}: {t.result}" for t in tasks if t.is_failed()]
        if errors:
            raise ValueError(
                f"{len(errors)} task{'s' if len(errors) != 1 else ''} failed: "
                + "\n".join(errors)
            )

    return [t.result for t in tasks]


async def run_tasks_async(
    tasks: list[Task],
    instructions: str = None,
    flow: Flow = None,
    agent: Agent = None,
    turn_strategy: TurnStrategy = None,
    raise_on_failure: bool = True,
    max_llm_calls: int = None,
    max_agent_turns: int = None,
    handlers: list[Union[Handler, AsyncHandler]] = None,
    model_kwargs: Optional[dict] = None,
    run_until: Optional[Union[RunEndCondition, Callable[[RunContext], bool]]] = None,
    stream: Union[bool, Stream] = False,
) -> Union[list[Any], AsyncIterator[tuple[Event, Any, Optional[Any]]]]:
    """
    Run a list of tasks asynchronously.
    """
    flow = flow or get_flow() or Flow()
    orchestrator = Orchestrator(
        tasks=tasks,
        flow=flow,
        agent=agent,
        turn_strategy=turn_strategy,
        handlers=handlers,
    )

    with controlflow.instructions(instructions):
        result = await orchestrator.run_async(
            max_llm_calls=max_llm_calls,
            max_agent_turns=max_agent_turns,
            model_kwargs=model_kwargs,
            run_until=run_until,
            stream=bool(stream),
        )

        if stream:
            # Convert True to ALL filter, otherwise use provided filter
            stream_filter = Stream.ALL if stream is True else stream
            return filter_events_async(result, stream_filter)

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
    model_kwargs: Optional[dict] = None,
    run_until: Optional[Union[RunEndCondition, Callable[[RunContext], bool]]] = None,
    stream: Union[bool, Stream] = False,
    **task_kwargs,
) -> Union[Any, Iterator[tuple[Event, Any, Optional[Any]]]]:
    """
    Run a single task.

    Args:
        objective: Objective of the task.
        turn_strategy: Turn strategy to use for the task.
        max_llm_calls: Maximum number of LLM calls to make.
        max_agent_turns: Maximum number of agent turns to make.
        raise_on_failure: Whether to raise an error if the task fails.
        handlers: List of handlers to use for the task.
        model_kwargs: Keyword arguments to pass to the LLM.
        run_until: Condition to stop running the task.
        stream: If True, stream all events. Can also provide StreamFilter flags to filter specific events.
               e.g. StreamFilter.CONTENT | StreamFilter.AGENT_TOOLS
    """
    task = Task(objective=objective, **task_kwargs)
    results = run_tasks(
        tasks=[task],
        raise_on_failure=raise_on_failure,
        turn_strategy=turn_strategy,
        max_llm_calls=max_llm_calls,
        max_agent_turns=max_agent_turns,
        handlers=handlers,
        model_kwargs=model_kwargs,
        run_until=run_until,
        stream=stream,
    )
    if stream:
        return results
    else:
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
    handlers: list[Union[Handler, AsyncHandler]] = None,
    model_kwargs: Optional[dict] = None,
    run_until: Optional[Union[RunEndCondition, Callable[[RunContext], bool]]] = None,
    stream: Union[bool, Stream] = False,
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
        model_kwargs=model_kwargs,
        run_until=run_until,
        stream=stream,
    )
    if stream:
        return results
    else:
        return results[0]
