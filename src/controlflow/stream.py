# Example usage
#
# # Stream all events
# for event in cf.stream.events("Write a story"):
#     print(event)
#
# # Stream just messages
# for event in cf.stream.events("Write a story", events='messages'):
#     print(event.content)
#
# # Stream just the result
# for delta, snapshot in cf.stream.result("Write a story"):
#     print(f"New: {delta}")
#
# # Stream results from multiple tasks
# for delta, snapshot in cf.stream.result_from_tasks([task1, task2]):
#     print(f"New result: {delta}")
#
from typing import Any, AsyncIterator, Callable, Iterator, Literal, Optional, Union

from controlflow.events.base import Event
from controlflow.events.events import (
    AgentMessage,
    AgentMessageDelta,
    AgentToolCall,
    ToolResult,
)
from controlflow.orchestration.handler import AsyncHandler, Handler
from controlflow.orchestration.orchestrator import Orchestrator
from controlflow.tasks.task import Task

StreamEvents = Union[list[str], Literal["all", "messages", "tools", "completion_tools"]]


def events(
    objective: str,
    *,
    events: StreamEvents = "all",
    filter_fn: Optional[Callable[[Event], bool]] = None,
    **kwargs,
) -> Iterator[Event]:
    """
    Stream events from a task execution.

    Args:
        objective: The task objective
        events: Which events to stream. Can be list of event types or:
               'all' - all events
               'messages' - agent messages
               'tools' - all tool calls/results
               'completion_tools' - only completion tools
        filter_fn: Optional additional filter function
        **kwargs: Additional arguments passed to Task

    Returns:
        Iterator of Event objects
    """

    def get_event_filter():
        if isinstance(events, list):
            return lambda e: e.event in events
        elif events == "messages":
            return lambda e: isinstance(e, (AgentMessage, AgentMessageDelta))
        elif events == "tools":
            return lambda e: isinstance(e, (AgentToolCall, ToolResult))
        elif events == "completion_tools":
            return lambda e: (
                isinstance(e, (AgentToolCall, ToolResult))
                and e.tool_call["name"].startswith("mark_task_")
            )
        else:  # 'all'
            return lambda e: True

    event_filter = get_event_filter()

    def event_handler(event: Event):
        if event_filter(event) and (not filter_fn or filter_fn(event)):
            yield event

    task = Task(objective=objective)
    task.run(handlers=[Handler(event_handler)], **kwargs)


def result(
    objective: str,
    **kwargs,
) -> Iterator[tuple[Any, Any]]:
    """
    Stream result from a task execution.

    Args:
        objective: The task objective
        **kwargs: Additional arguments passed to Task

    Returns:
        Iterator of (delta, accumulated) result tuples
    """
    current_result = None

    def result_handler(event: Event):
        nonlocal current_result
        if isinstance(event, ToolResult):
            if event.tool_call["name"].startswith("mark_task_"):
                result = event.tool_result.result  # Get actual result value
                if result != current_result:  # Only yield if changed
                    current_result = result
                    yield (result, result)  # For now delta == full result

    task = Task(objective=objective)
    task.run(handlers=[Handler(result_handler)], **kwargs)


def events_from_tasks(
    tasks: list[Task],
    events: StreamEvents = "all",
    filter_fn: Optional[Callable[[Event], bool]] = None,
    **kwargs,
) -> Iterator[Event]:
    """Stream events from multiple task executions."""

    def get_event_filter():
        if isinstance(events, list):
            return lambda e: e.event in events
        elif events == "messages":
            return lambda e: isinstance(e, (AgentMessage, AgentMessageDelta))
        elif events == "tools":
            return lambda e: isinstance(e, (AgentToolCall, ToolResult))
        elif events == "completion_tools":
            return lambda e: (
                isinstance(e, (AgentToolCall, ToolResult))
                and e.tool_call["name"].startswith("mark_task_")
            )
        else:  # 'all'
            return lambda e: True

    event_filter = get_event_filter()

    def event_handler(event: Event):
        if event_filter(event) and (not filter_fn or filter_fn(event)):
            yield event

    orchestrator = Orchestrator(
        tasks=tasks, handlers=[Handler(event_handler)], **kwargs
    )
    orchestrator.run()


def result_from_tasks(
    tasks: list[Task],
    **kwargs,
) -> Iterator[tuple[Any, Any]]:
    """Stream results from multiple task executions."""
    current_results = {task.id: None for task in tasks}

    def result_handler(event: Event):
        if isinstance(event, ToolResult):
            if event.tool_call["name"].startswith("mark_task_"):
                task_id = event.task.id
                result = event.tool_result.result
                if result != current_results[task_id]:
                    current_results[task_id] = result
                    yield (result, result)

    orchestrator = Orchestrator(
        tasks=tasks, handlers=[Handler(result_handler)], **kwargs
    )
    orchestrator.run()
