import datetime
import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import ConfigDict, Field
from pydantic_extra_types.pendulum_dt import DateTime

from controlflow.utilities.general import ControlFlowModel

if TYPE_CHECKING:
    from controlflow.events.message_compiler import CompileContext
    from controlflow.llm.messages import BaseMessage

# This is a global variable that will be shared between all instances of InMemoryStore
IN_MEMORY_STORE = {}


class Event(ControlFlowModel):
    model_config: ConfigDict = ConfigDict(extra="forbid")

    event: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: Optional[str] = None
    timestamp: DateTime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    persist: bool = True

    def to_messages(self, context: "CompileContext") -> list["BaseMessage"]:
        return []

    def __repr__(self) -> str:
        return f"<Event: {self.event} Timestamp: {self.timestamp}>"


class UnpersistedEvent(Event):
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
    persist: bool = False
