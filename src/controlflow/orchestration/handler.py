from typing import TYPE_CHECKING, Callable

from controlflow.events.base import Event

if TYPE_CHECKING:
    from controlflow.events.events import (
        AgentMessage,
        AgentMessageDelta,
        EndTurn,
        OrchestratorMessage,
        ToolCallEvent,
        ToolResultEvent,
        UserMessage,
    )
    from controlflow.events.orchestrator_events import (
        OrchestratorEnd,
        OrchestratorError,
        OrchestratorStart,
    )


class Handler:
    def handle(self, event: Event):
        """
        Handle is called whenever an event is emitted.

        By default, it dispatches to a method named after the event type e.g.
        `self.on_{event_type}(event=event)`.

        The `on_event` method is always called for every event.
        """
        self.on_event(event=event)
        event_type = event.event.replace("-", "_")
        method = getattr(self, f"on_{event_type}", None)
        if method:
            method(event=event)

    def on_event(self, event: Event):
        pass

    def on_orchestrator_start(self, event: "OrchestratorStart"):
        pass

    def on_orchestrator_end(self, event: "OrchestratorEnd"):
        pass

    def on_orchestrator_error(self, event: "OrchestratorError"):
        pass

    def on_agent_message(self, event: "AgentMessage"):
        pass

    def on_agent_message_delta(self, event: "AgentMessageDelta"):
        pass

    def on_tool_call(self, event: "ToolCallEvent"):
        pass

    def on_tool_result(self, event: "ToolResultEvent"):
        pass

    def on_orchestrator_message(self, event: "OrchestratorMessage"):
        pass

    def on_user_message(self, event: "UserMessage"):
        pass

    def on_end_turn(self, event: "EndTurn"):
        pass


class CallbackHandler(Handler):
    def __init__(self, callback: Callable[[Event], None]):
        self.callback = callback

    def on_event(self, event: Event):
        self.callback(event)
