from enum import Flag, auto
from typing import Any, AsyncIterator, Iterator, Optional, Union

from controlflow.events.events import (
    AgentContent,
    AgentContentDelta,
    AgentMessage,
    AgentMessageDelta,
    AgentToolCall,
    AgentToolCallDelta,
    Event,
    ToolResult,
)
from controlflow.events.task_events import (
    TaskFailure,
    TaskSkipped,
    TaskStart,
    TaskSuccess,
)


class Stream(Flag):
    """
    Filter flags for event streaming.

    Can be combined using bitwise operators:
    stream_filter = Stream.CONTENT | Stream.AGENT_TOOLS
    """

    NONE = 0
    ALL = auto()  # All events
    CONTENT = auto()  # Agent content and deltas
    AGENT_TOOLS = auto()  # Non-completion tool events
    COMPLETION_TOOLS = auto()  # Completion tool events
    TOOLS = AGENT_TOOLS | COMPLETION_TOOLS  # All tool events
    TASK_EVENTS = auto()  # Task state change events


def should_include_event(event: Event, stream_filter: Stream) -> bool:
    """Determine if an event should be included based on the stream filter."""
    # Pass all events if ALL is specified
    if stream_filter == Stream.ALL:
        return True

    # Content events
    if isinstance(event, (AgentContent, AgentContentDelta)):
        return bool(stream_filter & Stream.CONTENT)

    # Tool events
    if isinstance(event, (AgentToolCall, AgentToolCallDelta, ToolResult)):
        if is_completion_tool_event(event):
            return bool(stream_filter & Stream.COMPLETION_TOOLS)
        return bool(stream_filter & Stream.AGENT_TOOLS)

    # Task events
    if isinstance(event, (TaskStart, TaskSuccess, TaskFailure, TaskSkipped)):
        return bool(stream_filter & Stream.TASK_EVENTS)

    return False


def is_completion_tool_event(event: Event) -> bool:
    """Check if an event is related to a completion tool call."""
    if isinstance(event, ToolResult):
        tool = event.tool_result.tool
    elif isinstance(event, (AgentToolCall, AgentToolCallDelta)):
        tool = event.tool
    else:
        return False

    return tool and tool.metadata.get("is_completion_tool")


def process_event(event: Event) -> tuple[Event, Any, Optional[Any]]:
    """Process a single event and return the appropriate tuple."""
    # Message events
    if isinstance(event, AgentMessage):
        return event, event.message, None
    elif isinstance(event, AgentMessageDelta):
        return event, event.message_snapshot, event.message_delta

    # Content events
    elif isinstance(event, AgentContent):
        return event, event.content, None
    elif isinstance(event, AgentContentDelta):
        return event, event.content_snapshot, event.content_delta

    # Tool call events
    elif isinstance(event, AgentToolCall):
        return event, event.tool_call, None
    elif isinstance(event, AgentToolCallDelta):
        return event, event.tool_call_snapshot, event.tool_call_delta

    # Tool result events
    elif isinstance(event, ToolResult):
        return event, event.tool_result, None

    else:
        # Pass through any other events with no snapshot/delta
        return event, None, None


def filter_events_sync(
    events: Iterator[Event], stream_filter: Stream
) -> Iterator[tuple[Event, Any, Optional[Any]]]:
    """Synchronously filter events based on Stream flags."""
    for event in events:
        if should_include_event(event, stream_filter):
            yield process_event(event)


async def filter_events_async(
    events: AsyncIterator[Event], stream_filter: Stream
) -> AsyncIterator[tuple[Event, Any, Optional[Any]]]:
    """Asynchronously filter events based on Stream flags."""
    async for event in events:
        if should_include_event(event, stream_filter):
            yield process_event(event)
