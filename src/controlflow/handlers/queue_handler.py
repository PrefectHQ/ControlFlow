"""
A handler that queues events in a queue.
"""

import asyncio
import queue
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from controlflow.events.base import Event
from controlflow.events.events import (
    AgentMessage,
    AgentMessageDelta,
    AgentToolCall,
    ToolResult,
)
from controlflow.orchestration.handler import AsyncHandler, Handler


class QueueHandler(Handler):
    def __init__(
        self, queue: queue.Queue = None, event_filter: Callable[[Event], bool] = None
    ):
        self.queue = queue or queue.Queue()
        self.event_filter = event_filter

    def on_event(self, event: Event):
        if self.event_filter and not self.event_filter(event):
            return
        self.queue.put(event)


class AsyncQueueHandler(AsyncHandler):
    def __init__(
        self, queue: asyncio.Queue = None, event_filter: Callable[[Event], bool] = None
    ):
        self.queue = queue or asyncio.Queue()
        self.event_filter = event_filter

    async def on_event(self, event: Event):
        if self.event_filter and not self.event_filter(event):
            return
        await self.queue.put(event)


def message_filter(event: Event) -> bool:
    return isinstance(event, (AgentMessage, AgentMessageDelta))


def tool_filter(event: Event) -> bool:
    return isinstance(event, (AgentToolCall, ToolResult))


def result_filter(event: Event) -> bool:
    return isinstance(event, (AgentToolCall, ToolResult)) and event.tool_call[
        "name"
    ].startswith("mark_task_")
