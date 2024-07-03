from controlflow.events.events import Event


class Handler:
    def on_event(self, event: Event):
        event_type = event.event.replace("-", "_")
        method = getattr(self, f"on_{event_type}", None)
        if method:
            method(event=event)
