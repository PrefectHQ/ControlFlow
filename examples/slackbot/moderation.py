from typing import Annotated, TypedDict

import marvin
from prefect.events import emit_event
from pydantic import Field

from .types import SlackEvent

Activation = Annotated[float, Field(ge=0, le=1)]


class ModerationException(Exception):
    """Exception raised when a message is not allowed."""

    ...


class ViolationActivation(TypedDict):
    """Violation activation."""

    extreme_profanity: Annotated[Activation, Field(description="hell / damn are fine")]
    sexually_explicit: Activation
    hate_speech: Activation
    harassment: Activation
    self_harm: Activation
    dangerous_content: Activation
    makes_any_reference_to_bears_in_central_park: bool  # for testing


def emit_moderated_event(event: SlackEvent, activation: ViolationActivation):
    """Emit an IO event."""
    if not event.user:
        return

    emit_event(
        "slackbot.event.moderated",
        resource={"prefect.resource.id": event.user},
        payload={
            "activation": activation,
            "event": event.model_dump(),
        },
    )


def moderate_event(event: SlackEvent) -> tuple[str, ...]:
    """Moderate an event."""
    assert (text := event.text) is not None, "No text found on event"
    assert (channel := event.channel) is not None, "No channel found on event"
    assert (
        thread_ts := event.thread_ts or event.ts
    ) is not None, "No thread_ts found on event"

    activation: ViolationActivation = marvin.cast(
        event.model_dump_json(include={"type", "text", "user", "channel"}),
        ViolationActivation,
    )

    emit_moderated_event(event, activation)

    if activation["extreme_profanity"] > 0.9:
        raise ModerationException("Message contains extreme profanity.")

    if activation["makes_any_reference_to_bears_in_central_park"]:
        raise ModerationException("where you going with that whale buddy?")

    return text, channel, thread_ts
