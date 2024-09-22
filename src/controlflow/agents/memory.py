import abc
import re
from typing import Dict, List, Self

from pydantic import Field, field_validator, model_validator

import controlflow
from controlflow.tools.tools import Tool
from controlflow.utilities.general import ControlFlowModel


def sanitize_memory_key(key: str) -> str:
    # Remove any characters that are not alphanumeric or underscore
    return re.sub(r"[^a-zA-Z0-9_]", "", key)


class MemoryProvider(ControlFlowModel, abc.ABC):
    def configure(self, memory_key: str) -> None:
        """Configure the provider for a specific memory."""
        pass

    @abc.abstractmethod
    def add(self, memory_key: str, content: str) -> str:
        """Create a new memory and return its ID."""
        pass

    @abc.abstractmethod
    def delete(self, memory_key: str, memory_id: str) -> None:
        """Delete a memory by its ID."""
        pass

    @abc.abstractmethod
    def search(self, memory_key: str, query: str, n: int = 20) -> Dict[str, str]:
        """Search for n memories using a string query."""
        pass


class Memory(ControlFlowModel):
    key: str
    instructions: str = Field(
        default="Use this memory to store and retrieve important information."
    )
    provider: MemoryProvider = Field(
        default_factory=lambda: controlflow.defaults.memory_provider
    )

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        sanitized = sanitize_memory_key(v)
        if sanitized != v:
            raise ValueError(
                "Memory key must contain only alphanumeric characters and underscores"
            )
        return sanitized

    @model_validator(mode="after")
    def _configure_provider(self) -> Self:
        self.provider.configure(self.key)
        return self

    def add(self, content: str) -> str:
        return self.provider.add(self.key, content)

    def delete(self, memory_id: str) -> None:
        self.provider.delete(self.key, memory_id)

    def search(self, query: str, n: int = 20) -> Dict[str, str]:
        return self.provider.search(self.key, query, n)

    def get_tools(self) -> List[Tool]:
        return [
            Tool.from_function(
                self.add,
                name=f"add_memory_{self.key}",
                description=f'Create a new memory in Memory: "{self.key}".',
            ),
            Tool.from_function(
                self.delete,
                name=f"delete_memory_{self.key}",
                description=f'Delete a memory by its ID from Memory: "{self.key}".',
            ),
            Tool.from_function(
                self.search,
                name=f"search_memories_{self.key}",
                description=f'Search for memories relevant to a string query in Memory: "{self.key}". Returns a dictionary of memory IDs and their contents.',
            ),
        ]
