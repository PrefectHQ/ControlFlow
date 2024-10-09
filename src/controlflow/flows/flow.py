import uuid
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Union

from prefect.context import FlowRunContext
from pydantic import Field, field_validator
from typing_extensions import Self

import controlflow
from controlflow.agents import Agent
from controlflow.events.base import Event
from controlflow.events.history import History
from controlflow.utilities.context import ctx
from controlflow.utilities.general import ControlFlowModel, unwrap
from controlflow.utilities.logging import get_logger
from controlflow.utilities.prefect import prefect_flow_context

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class Flow(ControlFlowModel):
    model_config = dict(arbitrary_types_allowed=True)
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: Optional[str] = None
    description: Optional[str] = None
    history: History = Field(
        default_factory=lambda: controlflow.defaults.history,
        description="An object for storing events that take place during the flow.",
    )
    tools: list[Callable] = Field(
        default_factory=list,
        description="Tools that will be available to every agent in the flow",
    )
    default_agent: Optional[Agent] = Field(
        None,
        description="The default agent for the flow. This agent will be used "
        "for any task that does not specify an agent.",
    )
    prompt: Optional[str] = Field(
        None, description="A prompt to display to the agent working on the flow."
    )
    parent: Optional["Flow"] = Field(
        None,
        description="The parent flow. This is the flow that created this flow.",
    )
    load_parent_events: bool = Field(
        True,
        description="Whether to load events from the parent flow. If a flow is nested, "
        "this will load events from the parent flow so that the child flow can "
        "access the full conversation history, even though the child flow is a separate thread.",
    )
    context: dict[str, Any] = {}
    _cm_stack: list[contextmanager] = []

    def __enter__(self) -> Self:
        # use stack so we can enter the context multiple times
        cm = self.create_context()
        self._cm_stack.append(cm)
        return cm.__enter__()

    def __exit__(self, *exc_info):
        # exit the context manager
        return self._cm_stack.pop().__exit__(*exc_info)

    def __init__(self, **kwargs):
        if kwargs.get("parent") is None:
            kwargs["parent"] = get_flow()
        super().__init__(**kwargs)

    @field_validator("description")
    def _validate_description(cls, v):
        if v:
            v = unwrap(v)
        return v

    def get_prompt(self) -> str:
        """
        Generate a prompt to share information about the flow with an agent.
        """
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.FlowTemplate(template=self.prompt, flow=self)
        return template.render()

    def get_events(
        self,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: Optional[int] = None,
        types: Optional[list[str]] = None,
    ) -> list[Event]:
        events = self.history.get_events(
            thread_id=self.thread_id,
            before_id=before_id,
            after_id=after_id,
            limit=limit,
            types=types,
        )

        if self.parent and self.load_parent_events:
            events.extend(
                self.parent.get_events(
                    before_id=before_id,
                    after_id=after_id,
                    limit=limit,
                    types=types,
                )
            )
        events = sorted(events, key=lambda x: x.timestamp)
        return events

    def add_events(self, events: list[Event]):
        for event in events:
            event.thread_id = self.thread_id
        self.history.add_events(thread_id=self.thread_id, events=events)

    @contextmanager
    def create_context(self, **prefect_kwargs) -> Generator[Self, None, None]:
        # create a new Prefect flow if we're not already in a flow run
        if FlowRunContext.get() is None:
            prefect_context = prefect_flow_context(**prefect_kwargs)
        else:
            prefect_context = nullcontext()

        with prefect_context:
            # creating a new flow will reset any parent task tracking
            with ctx(flow=self, tasks=None):
                yield self


def get_flow() -> Optional[Flow]:
    """
    Loads the flow from the context or returns a new
    """
    flow: Union[Flow, None] = ctx.get("flow")
    return flow


def get_flow_events(limit: int = None) -> list[Event]:
    """
    Loads events from the active flow's thread.
    """
    if limit is None:
        limit = 50
    flow = get_flow()
    if flow:
        return flow.get_events(limit=limit)
    else:
        return []
