from typing import Callable

from controlflow.events.base import Event


class Handler:
    def handle(self, event: Event):
        event_type = event.event.replace("-", "_")
        method = getattr(self, f"on_{event_type}", None)
        if method:
            method(event=event)


class CallbackHandler(Handler):
    def __init__(self, callback: Callable[[Event], None]):
        self.callback = callback

    def handle(self, event: Event):
        self.callback(event)
