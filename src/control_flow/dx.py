import functools
import inspect
from typing import Callable, TypeVar

from prefect import flow as prefect_flow
from prefect import task as prefect_task

from control_flow.core.agent import Agent
from control_flow.core.flow import Flow
from control_flow.core.task import Task, TaskStatus
from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger
from control_flow.utilities.marvin import patch_marvin
from control_flow.utilities.types import AssistantTool, Thread

logger = get_logger(__name__)
T = TypeVar("T")
NOT_PROVIDED = object()


def ai_flow(
    fn=None,
    *,
    thread: Thread = None,
    tools: list[AssistantTool | Callable] = None,
    model: str = None,
):
    """
    Prepare a function to be executed as a Control Flow flow.
    """

    if fn is None:
        return functools.partial(
            ai_flow,
            thread=thread,
            tools=tools,
            model=model,
        )

    @functools.wraps(fn)
    def wrapper(
        *args,
        flow_kwargs: dict = None,
        **kwargs,
    ):
        p_fn = prefect_flow(fn)

        flow_obj = Flow(
            **{
                "thread": thread,
                "tools": tools or [],
                "model": model,
                **(flow_kwargs or {}),
            }
        )

        logger.info(
            f'Executing AI flow "{fn.__name__}" on thread "{flow_obj.thread.id}"'
        )

        with ctx(flow=flow_obj), patch_marvin():
            return p_fn(*args, **kwargs)

    return wrapper


def ai_task(
    fn=None,
    *,
    objective: str = None,
    agents: list[Agent] = None,
    tools: list[AssistantTool | Callable] = None,
    user_access: bool = None,
):
    """
    Use a Python function to create an AI task. When the function is called, an
    agent is created to complete the task and return the result.
    """

    if fn is None:
        return functools.partial(
            ai_task,
            objective=objective,
            agents=agents,
            tools=tools,
            user_access=user_access,
        )

    sig = inspect.signature(fn)

    if objective is None:
        if fn.__doc__:
            objective = f"{fn.__name__}: {fn.__doc__}"
        else:
            objective = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, _agents: list[Agent] = None, **kwargs):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        task = Task(
            objective=objective,
            agents=_agents or agents,
            context=bound.arguments,
            result_type=fn.__annotations__.get("return"),
            user_access=user_access or False,
            tools=tools or [],
        )

        task.run()
        return task.result

    return wrapper


def _name_from_objective():
    """Helper function for naming task runs"""
    from prefect.runtime import task_run

    objective = task_run.parameters.get("task")

    if not objective:
        objective = "Follow general instructions"
    if len(objective) > 75:
        return f"Task: {objective[:75]}..."
    return f"Task: {objective}"


@prefect_task(task_run_name=_name_from_objective)
def run_ai(
    tasks: str | list[str],
    agents: list[Agent] = None,
    cast: T = NOT_PROVIDED,
    context: dict = None,
    tools: list[AssistantTool | Callable] = None,
    user_access: bool = False,
) -> T | list[T]:
    """
    Create and run an agent to complete a task with the given objective and
    context. This function is similar to an inline version of the @ai_task
    decorator.

    This inline version is useful when you want to create and run an ad-hoc AI
    task, without defining a function or using decorator syntax. It provides
    more flexibility in terms of dynamically setting the task parameters.
    Additional detail can be provided as `context`.
    """

    single_result = False
    if isinstance(tasks, str):
        single_result = True

        tasks = [tasks]

    if cast is NOT_PROVIDED:
        if not tasks:
            cast = None
        else:
            cast = str

    # load flow
    flow = ctx.get("flow", None)

    # create tasks
    if tasks:
        ai_tasks = [
            Task(
                objective=t,
                context=context or {},
                user_access=user_access or False,
                tools=tools or [],
            )
            for t in tasks
        ]
    else:
        ai_tasks = []

    # create agent
    if agents is None:
        agents = [Agent(user_access=user_access or False)]

    # create Controller
    from control_flow.core.controller.controller import Controller

    controller = Controller(tasks=ai_tasks, agents=agents, flow=flow)
    controller.run()

    if ai_tasks:
        if all(task.status == TaskStatus.SUCCESSFUL for task in ai_tasks):
            result = [task.result for task in ai_tasks]
            if single_result:
                result = result[0]
            return result
        elif failed_tasks := [
            task for task in ai_tasks if task.status == TaskStatus.FAILED
        ]:
            raise ValueError(
                f'Failed tasks: {", ".join([task.objective for task in failed_tasks])}'
            )
