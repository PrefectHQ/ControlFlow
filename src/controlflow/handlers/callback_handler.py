"""
A handler that calls a callback function for each event.
"""

from typing import TYPE_CHECKING, Any, Callable, Coroutine

from controlflow.events.base import Event
from controlflow.orchestration.handler import AsyncHandler, Handler


class CallbackHandler(Handler):
    def __init__(self, callback: Callable[[Event], None]):
        self.callback = callback

    def on_event(self, event: Event):
        self.callback(event)


class AsyncCallbackHandler(AsyncHandler):
    def __init__(self, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        self.callback = callback

    async def on_event(self, event: Event):
        await self.callback(event)
