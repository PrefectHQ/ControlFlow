import functools
import inspect
from typing import Any, Callable, Optional, Union

import controlflow
from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.tasks.task import Task
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import prefect_flow as prefect_flow
from controlflow.utilities.prefect import prefect_task as prefect_task

# from controlflow.utilities.marvin import patch_marvin
from controlflow.utilities.tasks import resolve_tasks

logger = get_logger(__name__)


def flow(
    fn: Optional[Callable[..., Any]] = None,
    *,
    thread: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[list[Callable[..., Any]]] = None,
    agents: Optional[list[Agent]] = None,
    lazy: Optional[bool] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[float, int]] = None,
    timeout_seconds: Optional[Union[float, int]] = None,
    kwargs: Optional[dict[str, Any]] = None,
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
        agents (list[Agent], optional): List of agents to be used in the flow. Defaults to None.
        lazy (bool, optional): Whether the flow should be run lazily. If not
            set, behavior is determined by the global `eager_mode` setting. Lazy execution means
            that tasks are not run and a `Flow` object is returned instead.

    Returns:
        callable: The wrapped function or a new flow decorator if `fn` is not provided.
    """
    ...

    if fn is None:
        return functools.partial(
            flow,
            thread=thread,
            instructions=instructions,
            tools=tools,
            agents=agents,
            lazy=lazy,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            kwargs=kwargs,
        )

    sig = inspect.signature(fn)

    # the flow decorator creates a proper prefect flow
    @prefect_flow(
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
        **kwargs or {},
    )
    @functools.wraps(fn)
    def wrapper(
        *args,
        flow_kwargs: dict = None,
        lazy_: bool = False,
        **kwargs,
    ):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        flow_kwargs = flow_kwargs or {}

        if thread is not None:
            flow_kwargs.setdefault("thread", thread)
        if tools is not None:
            flow_kwargs.setdefault("tools", tools)
        if agents is not None:
            flow_kwargs.setdefault("agents", agents)

        flow_obj = Flow(
            name=fn.__name__,
            description=fn.__doc__,
            context=bound.arguments,
            **flow_kwargs,
        )

        with flow_obj.create_context(create_prefect_flow_context=False):
            with controlflow.instructions(instructions):
                result = fn(*args, **kwargs)

                # Determine if we should run eagerly or lazily
                if lazy_ is not None:
                    run_eagerly = not lazy_
                elif lazy is not None:
                    run_eagerly = not lazy
                else:
                    run_eagerly = controlflow.settings.eager_mode

                if run_eagerly:
                    flow_obj.run()

                    # resolve any returned tasks; this will raise on failure
                    return resolve_tasks(result)
                else:
                    return flow_obj

    if lazy is True or (lazy is None and not controlflow.settings.eager_mode):
        wrapper.__annotations__["return"] = Flow

    return wrapper


def task(
    fn: Optional[Callable[..., Any]] = None,
    *,
    objective: Optional[str] = None,
    instructions: Optional[str] = None,
    agents: Optional[list["Agent"]] = None,
    tools: Optional[list[Callable[..., Any]]] = None,
    user_access: Optional[bool] = None,
    lazy: Optional[bool] = None,
    retries: Optional[int] = None,
    retry_delay_seconds: Optional[Union[float, int]] = None,
    timeout_seconds: Optional[Union[float, int]] = None,
    **task_kwargs: Optional[dict[str, Any]],
):
    """
    A decorator that turns a Python function into a Task. The Task objective is
    set to the function name, and the instructions are set to the function
    docstring. When the function is called in eager mode (default), the arguments are
    provided to the task as context, and the task is run to completion. If
    successful, the task result is returned; if failed, an error is raised. When
    the function is called with eager mode disabled or `lazy=True`, a Task object is
    returned which can be run later.

    Args:
        fn (callable, optional): The function to be wrapped as a task. If not provided,
            the decorator will act as a partial function and return a new task decorator.
        objective (str, optional): The objective of the task. Defaults to None, in which
            case the function name is used as the objective.
        instructions (str, optional): Instructions for the task. Defaults to None, in which
            case the function docstring is used as the instructions.
        agents (list[Agent], optional): List of agents to be used in the task. Defaults to None.
        tools (list[Callable], optional): List of tools to be used in the task. Defaults to None.
        user_access (bool, optional): Whether the task requires user access. Defaults to None,
            in which case it is set to False.
        lazy (bool, optional): Whether the task should be run lazily. If not
            set, behavior is determined by the global `eager_mode` setting. Lazy
            execution means that a `Task` object is returned instead of running the task.

    Returns:
        callable: The wrapped function or a new task decorator if `fn` is not provided.
    """

    if fn is None:
        return functools.partial(
            task,
            objective=objective,
            instructions=instructions,
            agents=agents,
            tools=tools,
            user_access=user_access,
            lazy=lazy,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            **task_kwargs,
        )

    sig = inspect.signature(fn)

    if objective is None:
        objective = fn.__name__

    if instructions is None:
        instructions = fn.__doc__

    result_type = fn.__annotations__.get("return")

    @prefect_task(
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    @functools.wraps(fn)
    def wrapper(*args, lazy_: bool = None, **kwargs):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        task = Task(
            objective=objective,
            instructions=instructions,
            agents=agents,
            context=bound.arguments,
            result_type=result_type,
            user_access=user_access or False,
            tools=tools or [],
            **task_kwargs,
        )

        # Determine if we should run eagerly or lazily
        if lazy_ is not None:
            run_eagerly = not lazy_
        elif lazy is not None:
            run_eagerly = not lazy
        else:
            run_eagerly = controlflow.settings.eager_mode

        if run_eagerly:
            task.run()
            return task.result
        else:
            return task

    if lazy is True or (lazy is None and not controlflow.settings.eager_mode):
        wrapper.__annotations__["return"] = Task

    return wrapper
