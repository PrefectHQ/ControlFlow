from enum import Flag, auto
from typing import Any, Iterator, Optional

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
    RESULTS = auto()  # Completion tool events
    TOOLS = AGENT_TOOLS | RESULTS  # All tool events


def filter_events(
    events: Iterator[Event], stream_filter: Stream
) -> Iterator[tuple[Event, Any, Optional[Any]]]:
    """
    Filter events based on Stream flags.
    Returns tuples of (event, snapshot, delta) where snapshot and delta depend on the event type.

    Returns:
        Iterator of (event, snapshot, delta) tuples where:
            - event: The original event
            - snapshot: Full state (e.g., complete message, tool state)
            - delta: Incremental change (None for non-delta events)

    Patterns for different event types:
        - Content events: (event, full_text, new_text)
        - Tool calls: (event, tool_state, tool_delta)
        - Tool results: (event, result_state, None)
        - Other events: (event, None, None)
    """

    def is_completion_tool_event(event: Event) -> bool:
        """Check if an event is related to a completion tool call."""
        if isinstance(event, ToolResult):
            tool = event.tool_result.tool
        elif isinstance(event, (AgentToolCall, AgentToolCallDelta)):
            tool = event.tool
        else:
            return False

        return tool and tool.metadata.get("is_completion_tool")

    def should_include_event(event: Event) -> bool:
        # Pass all events if ALL is specified
        if stream_filter == Stream.ALL:
            return True

        # Content events
        if isinstance(event, (AgentContent, AgentContentDelta)):
            return bool(stream_filter & Stream.CONTENT)

        # Tool events
        if isinstance(event, (AgentToolCall, AgentToolCallDelta, ToolResult)):
            if is_completion_tool_event(event):
                return bool(stream_filter & Stream.RESULTS)
            return bool(stream_filter & Stream.AGENT_TOOLS)

        return False

    for event in events:
        if not should_include_event(event):
            continue

        # Message events
        if isinstance(event, AgentMessage):
            yield event, event.message, None
        elif isinstance(event, AgentMessageDelta):
            yield event, event.message_snapshot, event.message_delta

        # Content events
        elif isinstance(event, AgentContent):
            yield event, event.content, None
        elif isinstance(event, AgentContentDelta):
            yield event, event.content_snapshot, event.content_delta

        # Tool call events
        elif isinstance(event, AgentToolCall):
            yield event, event.tool_call, None
        elif isinstance(event, AgentToolCallDelta):
            yield event, event.tool_call_snapshot, event.tool_call_delta

        # Tool result events
        elif isinstance(event, ToolResult):
            yield event, event.tool_result, None
        else:
            # Pass through any other events with no snapshot/delta
            yield event, None, None
