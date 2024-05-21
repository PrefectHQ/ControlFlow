import abc
import json
from pathlib import Path

from pydantic import Field, field_validator

import controlflow
from controlflow.utilities.types import ControlFlowModel, Message

_IN_MEMORY_HISTORY = dict()


class BaseHistory(ControlFlowModel, abc.ABC):
    @abc.abstractmethod
    def load_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        raise NotImplementedError()

    @abc.abstractmethod
    def save_messages(self, thread_id: str, messages: list[Message]):
        raise NotImplementedError()


class InMemoryHistory(BaseHistory):
    def load_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        return _IN_MEMORY_HISTORY.get(thread_id)[-limit:]

    def save_messages(self, thread_id: str, messages: list[Message]):
        _IN_MEMORY_HISTORY.setdefault(thread_id, []).extend(messages)


class FileHistory(BaseHistory):
    base_path: Path = Field(
        default_factory=lambda: controlflow.settings.home_path / "history"
    )

    def path(self, thread_id: str) -> Path:
        return self.base_path / f"{thread_id}.json"

    @field_validator("base_path", mode="before")
    def _validate_path(cls, v):
        v = Path(v).expanduser()
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    def load_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        if not self.path(thread_id).exists():
            return []
        with open(self.path(thread_id), "r") as f:
            all_messages = json.load(f)
            return [Message.model_validate(msg) for msg in all_messages[-limit:]]

    def save_messages(self, thread_id: str, messages: list[Message]):
        if self.path(thread_id).exists():
            with open(self.path(thread_id), "r") as f:
                all_messages = json.load(f)
        else:
            all_messages = []
        all_messages.extend([msg.model_dump(mode="json") for msg in messages])
        with open(self.path(thread_id), "w") as f:
            json.dump(all_messages, f)
