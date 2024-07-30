import datetime
import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from controlflow.utilities.general import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.events.message_compiler import CompileContext
    from controlflow.llm.messages import BaseMessage

# This is a global variable that will be shared between all instances of InMemoryStore
IN_MEMORY_STORE = {}


class Event(ControlFlowModel):
    model_config: dict = dict(extra="forbid")

    event: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: Optional[str] = None
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    persist: bool = True

    def to_messages(self, context: "CompileContext") -> list["BaseMessage"]:
        return []


class UnpersistedEvent(Event):
    model_config = dict(arbitrary_types_allowed=True)
    persist: bool = False
