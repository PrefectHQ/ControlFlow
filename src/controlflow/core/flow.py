import functools
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Union

import prefect
from marvin.beta.assistants import Thread
from openai.types.beta.threads import Message
from pydantic import Field, field_validator

import controlflow
from controlflow.utilities.context import ctx
from controlflow.utilities.logging import get_logger
from controlflow.utilities.marvin import patch_marvin
from controlflow.utilities.types import ControlFlowModel, ToolType

if TYPE_CHECKING:
    from controlflow.core.agent import Agent
    from controlflow.core.task import Task
logger = get_logger(__name__)


class Flow(ControlFlowModel):
    thread: Thread = Field(None, validate_default=True)
    tools: list[ToolType] = Field(
        default_factory=list,
        description="Tools that will be available to every agent in the flow",
    )
    agents: list["Agent"] = Field(
        description="The default agents for the flow. These agents will be used "
        "for any task that does not specify agents.",
        default_factory=list,
    )
    _tasks: dict[str, "Task"] = {}
    context: dict[str, Any] = {}

    @field_validator("thread", mode="before")
    def _load_thread_from_ctx(cls, v):
        if v is None:
            v = ctx.get("thread", None)
            if v is None:
                v = Thread()
        if not v.id:
            v.create()

        return v

    def add_task(self, task: "Task"):
        if self._tasks.get(task.id, task) is not task:
            raise ValueError(
                f"A different task with id '{task.id}' already exists in flow."
            )
        self._tasks[task.id] = task

    @contextmanager
    def _context(self):
        with ctx(flow=self, tasks=[]):
            yield self

    def __enter__(self):
        self.__cm = self._context()
        return self.__cm.__enter__()

    def __exit__(self, *exc_info):
        return self.__cm.__exit__(*exc_info)


GLOBAL_FLOW = None


def get_flow() -> Flow:
    """
    Loads the flow from the context.

    Will error if no flow is found in the context, unless the global flow is
    enabled in settings
    """
    flow: Union[Flow, None] = ctx.get("flow")
    if not flow:
        if controlflow.settings.enable_global_flow:
            return GLOBAL_FLOW
        else:
            raise ValueError("No flow found in context.")
    return flow


def reset_global_flow():
    global GLOBAL_FLOW
    GLOBAL_FLOW = Flow()


def get_flow_messages(limit: int = None) -> list[Message]:
    """
    Loads messages from the flow's thread.

    Will error if no flow is found in the context.
    """
    flow = get_flow()
    return flow.thread.get_messages(limit=limit)


def flow(
    fn=None,
    *,
    thread: Thread = None,
    instructions: str = None,
    tools: list[ToolType] = None,
    agents: list["Agent"] = None,
):
    """
    A decorator that runs a function as a Flow
    """

    if fn is None:
        return functools.partial(
            flow,
            thread=thread,
            tools=tools,
            agents=agents,
        )

    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(
        *args,
        flow_kwargs: dict = None,
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

        p_fn = prefect.flow(fn)

        flow_obj = Flow(**flow_kwargs, context=bound.arguments)

        logger.info(
            f'Executing AI flow "{fn.__name__}" on thread "{flow_obj.thread.id}"'
        )

        with ctx(flow=flow_obj), patch_marvin():
            with controlflow.instructions(instructions):
                return p_fn(*args, **kwargs)

    return wrapper
