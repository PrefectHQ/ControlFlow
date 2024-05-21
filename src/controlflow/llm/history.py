import uuid

from pydantic import Field

from controlflow.utilities.types import ControlFlowModel, Message


class Thread(ControlFlowModel):
    id: str = Field(default_factory=uuid.uuid4().hex[:8])


class History(ControlFlowModel):
    thread: Thread
    messages: list[Message]
