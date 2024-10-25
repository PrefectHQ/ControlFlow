import asyncio
import functools
import inspect
from typing import Any, Callable, Optional, Union

from prefect.utilities.asyncutils import run_coro_as_sync

import controlflow
from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.tasks.task import Task
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import prefect_flow, prefect_task

# from controlflow.utilities.marvin import patch_marvin

logger = get_logger(__name__)


def flow(
    fn: Optional[Callable[..., Any]] = None,
    *,
    thread: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[list[Callable[..., Any]]] = None,
    default_agent: Optional[Agent] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[float, int]] = None,
    timeout_seconds: Optional[Union[float, int]] = None,
    prefect_kwargs: Optional[dict[str, Any]] = None,
    context_kwargs: Optional[list[str]] = None,
    **kwargs: Optional[dict[str, Any]],
):
    """
    A decorator that wraps a function as a ControlFlow flow.

    When the function is called, a new flow is created and any tasks created
    within the function will be run as part of that flow. When the function
    returns, all tasks created in the flow will be run to completion (if they
    were not already completed) and their results will be returned. Any tasks
    that are returned from the function will be replaced with their resolved
    result.

    Args:
        fn (callable, optional): The function to be wrapped as a flow. If not provided,
            the decorator will act as a partial function and return a new flow decorator.
        thread (str, optional): The thread to execute the flow on. Defaults to None.
        instructions (str, optional): Instructions for the flow. Defaults to None.
        tools (list[Callable], optional): List of tools to be used in the flow. Defaults to None.
        default_agent (Agent, optional): The default agent to be used in the flow. Defaults to None.
        context_kwargs (list[str], optional): List of argument names to be added to the flow context.
            Defaults to None.
    Returns:
        callable: The wrapped function or a new flow decorator if `fn` is not provided.
    """
    if fn is None:
        return functools.partial(
            flow,
            thread=thread,
            instructions=instructions,
            tools=tools,
            default_agent=default_agent,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            context_kwargs=context_kwargs,
            **kwargs,
        )

    sig = inspect.signature(fn)

    def create_flow_context(bound_args):
        flow_kwargs = kwargs.copy()
        if thread is not None:
            flow_kwargs.setdefault("thread_id", thread)
        if tools is not None:
            flow_kwargs.setdefault("tools", tools)
        if default_agent is not None:
            flow_kwargs.setdefault("default_agent", default_agent)

        context = {}
        if context_kwargs:
            context = {k: bound_args[k] for k in context_kwargs if k in bound_args}

        return Flow(
            name=fn.__name__,
            description=fn.__doc__,
            context=context,
            **flow_kwargs,
        )

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*wrapper_args, **wrapper_kwargs):
            bound = sig.bind(*wrapper_args, **wrapper_kwargs)
            bound.apply_defaults()
            with (
                create_flow_context(bound.arguments),
                controlflow.instructions(instructions),
            ):
                return await fn(*wrapper_args, **wrapper_kwargs)
    else:

        @functools.wraps(fn)
        def wrapper(*wrapper_args, **wrapper_kwargs):
            bound = sig.bind(*wrapper_args, **wrapper_kwargs)
            bound.apply_defaults()
            with (
                create_flow_context(bound.arguments),
                controlflow.instructions(instructions),
            ):
                return fn(*wrapper_args, **wrapper_kwargs)

    wrapper = prefect_flow(
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        **(prefect_kwargs or {}),
    )(wrapper)
    return wrapper


def task(
    fn: Optional[Callable[..., Any]] = None,
    *,
    objective: Optional[str] = None,
    instructions: Optional[str] = None,
    name: Optional[str] = None,
    agents: Optional[list["Agent"]] = None,
    tools: Optional[list[Callable[..., Any]]] = None,
    interactive: Optional[bool] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[float, int]] = None,
    timeout_seconds: Optional[Union[float, int]] = None,
    **task_kwargs: Optional[dict[str, Any]],
):
    """
    A decorator that turns a Python function into a Task. The Task objective is
    set to the function name, and the instructions are set to the function
    docstring. When the function is called, the arguments are provided to the
    task as context, and the task is run to completion. If successful, the task
    result is returned; if failed, an error is raised.

    Args:
        fn (callable, optional): The function to be wrapped as a task. If not provided,
            the decorator will act as a partial function and return a new task decorator.
        objective (str, optional): The objective of the task. Defaults to None, in which
            case the function name is used as the objective.
        instructions (str, optional): Instructions for the task. Defaults to None, in which
            case the function docstring is used as the instructions.
        agents (list[Agent], optional): List of agents to be used in the task. Defaults to None.
        tools (list[Callable], optional): List of tools to be used in the task. Defaults to None.
        interactive (bool, optional): Whether the task requires human interaction or input during its execution. Defaults to None, in which case it is set to False.

    Returns:
        callable: The wrapped function or a new task decorator if `fn` is not provided.
    """

    if fn is None:
        return functools.partial(
            task,
            objective=objective,
            instructions=instructions,
            name=name,
            agents=agents,
            tools=tools,
            interactive=interactive,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            **task_kwargs,
        )

    sig = inspect.signature(fn)

    if name is None:
        name = fn.__name__

    if objective is None:
        objective = fn.__doc__ or ""

    result_type = fn.__annotations__.get("return")

    def _get_task(*args, **kwargs) -> Task:
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        context = bound.arguments.copy()

        # call the function to see if it produces an updated objective
        maybe_coro = fn(*args, **kwargs)
        if asyncio.iscoroutine(maybe_coro):
            result = run_coro_as_sync(maybe_coro)
        else:
            result = maybe_coro
        if result is not None:
            context["Additional context"] = result

        return Task(
            objective=objective,
            instructions=instructions,
            name=name,
            agents=agents,
            context=context,
            result_type=result_type,
            interactive=interactive or False,
            tools=tools or [],
            **task_kwargs,
        )

    if asyncio.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            task = _get_task(*args, **kwargs)
            return await task.run_async()
    else:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            task = _get_task(*args, **kwargs)
            return task.run()

    wrapper = prefect_task(
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )(wrapper)

    # store the `as_task` method for loading the task object
    wrapper.as_task = _get_task

    return wrapper
