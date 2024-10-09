import abc
import json
import math
from functools import cache
from pathlib import Path
from typing import Optional, Union

from pydantic import Field, TypeAdapter, field_validator

import controlflow
from controlflow.events.base import Event
from controlflow.utilities.general import ControlFlowModel

# This is a global variable that will be shared between all instances of InMemoryStore
IN_MEMORY_STORE = {}


@cache
def get_event_validator() -> TypeAdapter:
    from controlflow.events.events import (
        AgentMessage,
        EndTurn,
        OrchestratorMessage,
        ToolResultEvent,
        UserMessage,
    )

    types = Union[
        OrchestratorMessage,
        UserMessage,
        AgentMessage,
        EndTurn,
        ToolResultEvent,
        Event,
    ]
    return TypeAdapter(list[types])


def filter_events(
    events: list[Event],
    types: Optional[list[str]] = None,
    before_id: Optional[str] = None,
    after_id: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Filters a list of events based on the specified criteria.

    Args:
        events (list[Event]): The list of events to filter.
        tags: (Optional[list[str]]): The tags to filter by. Defaults to None.
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
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        """
        Retrieve a list of events based on the specified criteria.

        Args:
            thread_id (str): The ID of the thread to retrieve events from.
            tags (Optional[list[str]]): The tags associated with the events (default: None).
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
            types=types,
            before_id=before_id,
            after_id=after_id,
            limit=limit,
        )


class FileHistory(History):
    base_path: Path = Field(
        default_factory=lambda: controlflow.settings.home_path / "history/FileHistory"
    )

    def path(self, thread_id: str) -> Path:
        return self.base_path / f"{thread_id}.json"

    def get_events(
        self,
        thread_id: str,
        types: Optional[list[str]] = None,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Event]:
        """
        Retrieves a list of events based on the specified criteria.

        Args:
            thread_id (str): The ID of the thread to retrieve events from.
            types (Optional[list[str]]): The list of event types to filter by (default: None).
            before_id (Optional[str]): The ID of the event before which to stop retrieving events (default: None).
            after_id (Optional[str]): The ID of the event after which to start retrieving events (default: None).
            limit (Optional[int]): The maximum number of events to retrieve (default: None).

        Returns:
            list[Event]: A list of events that match the specified criteria.
        """
        file_path = self.path(thread_id)

        if not file_path.exists():
            return []

        with file_path.open("r") as f:
            raw_data = f.read()

        validator = get_event_validator()
        events = validator.validate_json(raw_data)

        return filter_events(
            events=events,
            types=types,
            before_id=before_id,
            after_id=after_id,
            limit=limit,
        )

    def add_events(self, thread_id: str, events: list[Event]):
        # TODO: this is pretty inefficient because we read / write the entire file
        # every time instead of doing it incrementally. Need to switch to JSONL
        # if we want to improve performance.
        file_path = self.path(thread_id)

        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

        with file_path.open("r") as f:
            try:
                all_events = json.load(f)
            except json.JSONDecodeError:
                all_events = []

        all_events.extend([event.model_dump(mode="json") for event in events])

        with file_path.open("w") as f:
            json.dump(all_events, f)
