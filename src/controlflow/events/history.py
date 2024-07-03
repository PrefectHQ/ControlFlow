import abc
import json
import math
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from pydantic import Field, TypeAdapter, field_validator

import controlflow
from controlflow.events.agent_events import (
    AgentMessageEvent,
    EndTurnEvent,
    SelectAgentEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from controlflow.events.events import Event
from controlflow.events.task_events import TaskCompleteEvent, TaskReadyEvent
from controlflow.events.tool_events import ToolResultEvent
from controlflow.utilities.types import ControlFlowModel

if TYPE_CHECKING:
    pass

# This is a global variable that will be shared between all instances of InMemoryStore
IN_MEMORY_STORE = {}


def get_default_history() -> "History":
    return controlflow.default_history


@cache
def get_event_validator() -> TypeAdapter:
    types = Union[
        TaskReadyEvent,
        TaskCompleteEvent,
        SelectAgentEvent,
        SystemMessageEvent,
        UserMessageEvent,
        AgentMessageEvent,
        EndTurnEvent,
        ToolResultEvent,
        Event,
    ]
    return TypeAdapter(list[types])


def filter_events(
    events: list[Event],
    agent_ids: Optional[list[str]] = None,
    task_ids: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
    before_id: Optional[str] = None,
    after_id: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Filters a list of events based on the specified criteria.

    Args:
        events (list[Event]): The list of events to filter.
        agent_ids (Optional[list[str]]): The agent ids to filter by. Defaults to None.
        task_ids (Optional[list[str]]): The task ids to filter by. Defaults to None.
        types (Optional[list[str]]): The event types to filter by. Defaults to None.
        before_id (Optional[str]): The ID of the event before which to start including events. Defaults to None.
        after_id (Optional[str]): The ID of the event after which to stop including events. Defaults to None.
        limit (Optional[int]): The maximum number of events to include. Defaults to None.

    Returns:
        list[Event]: The filtered list of events.
    """
    new_events = []
    seen_before_id = True if not before_id else False
    seen_after_id = False if not after_id else True

    for event in reversed(events):
        if event.id == before_id:
            seen_before_id = True
        if event.id == after_id:
            seen_after_id = True

        # if we haven't reached the `before_id` we can skip this event
        if not seen_before_id:
            continue

        # if we've reached the `after_id` we can stop searching
        if seen_after_id:
            break

        # if types are specified and this event is not one of them, skip it
        if types and event.event not in types:
            continue

        # if agent_ids are specified and this event has agent_ids and none of them are in the list, skip it
        agent_match = (
            (
                agent_ids
                and event.agent_ids
                and any(a in event.agent_ids for a in agent_ids)
            )
            or not agent_ids
            or not event.agent_ids
        )

        # if task_ids are specified and this event has task_ids and none of them are in the list, skip it
        task_match = (
            (task_ids and event.task_ids and any(t in event.task_ids for t in task_ids))
            or not task_ids
            or not event.task_ids
        )

        # if neither agent_ids nor task_ids were matched
        if not (agent_match or task_match):
            continue

        new_events.append(event)

        if len(new_events) >= (limit or math.inf):
            break

    return list(reversed(new_events))


class History(ControlFlowModel, abc.ABC):
    @abc.abstractmethod
    def get_events(
        self,
        thread_id: str,
        types: Optional[list[str]] = None,
        agent_ids: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        raise NotImplementedError()

    @abc.abstractmethod
    def add_events(self, thread_id: str, events: list[Event]):
        raise NotImplementedError()


class InMemoryHistory(History):
    history: dict[str, list[Event]] = Field(
        default_factory=lambda: IN_MEMORY_STORE, repr=False
    )

    def add_events(self, thread_id: str, events: list[Event]):
        self.history.setdefault(thread_id, []).extend(events)

    def get_events(
        self,
        thread_id: str,
        types: Optional[list[str]] = None,
        agent_ids: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        """
        Retrieve a list of events based on the specified criteria.

        Args:
            thread_id (str): The ID of the thread to retrieve events from.
            agent_ids (Optional[list[str]]): The agent associated with the events (default: None).
            task_ids (Optional[list[str]]): The list of tasks associated with the events (default: None).
            types (Optional[list[str]]): The list of event types to filter by (default: None).
            before_id (Optional[str]): The ID of the event before which to start retrieving events (default: None).
            after_id (Optional[str]): The ID of the event after which to stop retrieving events (default: None).
            limit (Optional[int]): The maximum number of events to retrieve (default: None).

        Returns:
            list[Event]: A list of events that match the specified criteria.

        """
        events = self.history.get(thread_id, [])
        return filter_events(
            events=events,
            agent_ids=agent_ids,
            task_ids=task_ids,
            types=types,
            before_id=before_id,
            after_id=after_id,
            limit=limit,
        )


class FileHistory(History):
    base_path: Path = Field(
        default_factory=lambda: controlflow.settings.home_path / "filestore_events"
    )

    def path(self, thread_id: str) -> Path:
        return self.base_path / f"{thread_id}.json"

    @field_validator("base_path", mode="before")
    def _validate_path(cls, v):
        v = Path(v).expanduser()
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    def get_events(
        self,
        thread_id: str,
        agent_ids: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        types: Optional[list[str]] = None,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        """
        Retrieves a list of events based on the specified criteria.

        Args:
            thread_id (str): The ID of the thread to retrieve events from.
            agent_ids (Optional[list[str]]): The agent associated with the events (default: None).
            task_ids (Optional[list[str]]): The list of tasks associated with the events (default: None).
            types (Optional[list[str]]): The list of event types to filter by (default: None).
            before_id (Optional[str]): The ID of the event before which to stop retrieving events (default: None).
            after_id (Optional[str]): The ID of the event after which to start retrieving events (default: None).
            limit (Optional[int]): The maximum number of events to retrieve (default: None).

        Returns:
            list[Event]: A list of events that match the specified criteria.
        """
        if not self.path(thread_id).exists():
            return []

        with open(self.path(thread_id), "r") as f:
            raw_data = f.read()

        validator = get_event_validator()
        events = validator.validate_json(raw_data)

        return filter_events(
            events=events,
            agent_ids=agent_ids,
            task_ids=task_ids,
            types=types,
            before_id=before_id,
            after_id=after_id,
            limit=limit,
        )

    def add_events(self, thread_id: str, events: list[Event]):
        if self.path(thread_id).exists():
            with open(self.path(thread_id), "r") as f:
                all_events = json.load(f)
        else:
            all_events = []
        all_events.extend([event.model_dump(mode="json") for event in events])
        with open(self.path(thread_id), "w") as f:
            json.dump(all_events, f)
