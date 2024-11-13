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
    stream_filter = StreamFilter.CONTENT | StreamFilter.AGENT_TOOLS
    """

    NONE = 0
    CONTENT = auto()  # Agent content and deltas
    AGENT_TOOLS = auto()  # Non-completion tool events
    RESULTS = auto()  # Completion tool events
    TOOLS = AGENT_TOOLS | RESULTS  # All tool events
    ALL = CONTENT | TOOLS  # Everything

    @classmethod
    def _filter_names(cls, filter: "Stream") -> list[str]:
        """Convert StreamFilter to list of filter names for filter_events()"""
        names = []
        if filter & cls.CONTENT:
            names.append("content")
        if filter & cls.AGENT_TOOLS:
            names.append("agent_tools")
        if filter & cls.RESULTS:
            names.append("results")
        return names


def filter_events(
    events: Iterator[Event], filters: list[str]
) -> Iterator[tuple[Event, Any, Optional[Any]]]:
    """
    Filter events based on a list of event types or shortcuts.
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

    def is_agent_tool_event(event: Event) -> bool:
        """Check if an event is related to a regular (non-completion) tool call."""
        if isinstance(event, ToolResult):
            tool = event.tool_result.tool
        elif isinstance(event, (AgentToolCall, AgentToolCallDelta)):
            tool = event.tool
        else:
            return False

        return tool and not tool.metadata.get("is_completion_tool")

    # Expand shortcuts to event types and build filtering predicates
    event_filters = []
    for filter_name in filters:
        if filter_name == "content":
            event_filters.append(
                lambda e: e.event in {"agent-content", "agent-content-delta"}
            )
        elif filter_name == "tools":
            event_filters.append(
                lambda e: e.event
                in {
                    "agent-tool-call",
                    "agent-tool-call-delta",
                    "tool-result",
                }
            )
        elif filter_name == "agent_tools":
            event_filters.append(
                lambda e: (
                    e.event
                    in {
                        "agent-tool-call",
                        "agent-tool-call-delta",
                        "tool-result",
                    }
                    and is_agent_tool_event(e)
                )
            )
        elif filter_name == "results":
            event_filters.append(
                lambda e: (
                    e.event
                    in {
                        "agent-tool-call",
                        "agent-tool-call-delta",
                        "tool-result",
                    }
                    and is_completion_tool_event(e)
                )
            )
        else:
            # Raw event type
            event_filters.append(lambda e, t=filter_name: e.event == t)

    def passes_filters(event: Event) -> bool:
        return any(f(event) for f in event_filters)

    for event in events:
        if not passes_filters(event):
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
            yield event, None, None
