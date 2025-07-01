import abc
import re
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

import controlflow
from controlflow.tools.tools import Tool
from controlflow.utilities.general import ControlFlowModel, unwrap
from controlflow.utilities.logging import get_logger

logger = get_logger("controlflow.memory")


def sanitize_memory_key(key: str) -> str:
    # Remove any characters that are not alphanumeric or underscore
    return re.sub(r"[^a-zA-Z0-9_]", "", key)


class AsyncMemoryProvider(ControlFlowModel, abc.ABC):
    async def configure(self, memory_key: str) -> None:
        """Configure the provider for a specific memory."""
        pass

    @abc.abstractmethod
    async def add(self, memory_key: str, content: str) -> str:
        """Create a new memory and return its ID."""
        pass

    @abc.abstractmethod
    async def delete(self, memory_key: str, memory_id: str) -> None:
        """Delete a memory by its ID."""
        pass

    @abc.abstractmethod
    async def search(self, memory_key: str, query: str, n: int = 20) -> Dict[str, str]:
        """Search for n memories using a string query."""
        pass


class AsyncMemory(ControlFlowModel):
    """
    A memory module is a partitioned collection of memories that are stored in a
    vector database, configured by a MemoryProvider.
    """

    key: str
    instructions: str = Field(
        description="Explain what this memory is for and how it should be used."
    )
    provider: AsyncMemoryProvider = Field(
        default_factory=lambda: controlflow.defaults.memory_provider,
        validate_default=True,
    )

    def __hash__(self) -> int:
        return id(self)

    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(
        cls, v: Optional[Union[AsyncMemoryProvider, str]]
    ) -> AsyncMemoryProvider:
        if isinstance(v, str):
            return get_memory_provider(v)
        if v is None:
            raise ValueError(
                unwrap(
                    """
                    Memory modules require a MemoryProvider to configure the
                    underlying vector database. No provider was passed as an
                    argument, and no default value has been configured. 
                    
                    For more information on configuring a memory provider, see
                    the [Memory
                    documentation](https://controlflow.ai/patterns/memory), and
                    please review the [default provider
                    guide](https://controlflow.ai/guides/default-memory) for
                    information on configuring a default provider.
                    
                    Please note that if you are using ControlFlow for the first
                    time, this error is expected because ControlFlow does not include
                    vector dependencies by default.
                    """
                )
            )
        return v

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        sanitized = sanitize_memory_key(v)
        if sanitized != v:
            raise ValueError(
                "Memory key must contain only alphanumeric characters and underscores"
            )
        return sanitized

    async def _configure_provider(self):
        await self.provider.configure(self.key)
        return self

    async def add(self, content: str) -> str:
        return await self.provider.add(self.key, content)

    async def delete(self, memory_id: str) -> None:
        await self.provider.delete(self.key, memory_id)

    async def search(self, query: str, n: int = 20) -> Dict[str, str]:
        return await self.provider.search(self.key, query, n)

    def get_tools(self) -> List[Tool]:
        return [
            Tool.from_function(
                self.add,
                name=f"store_memory_{self.key}",
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


def get_memory_provider(provider: str) -> AsyncMemoryProvider:
    logger.debug(f"Loading memory provider: {provider}")

    # --- async postgres ---

    if provider.startswith("async-postgres"):
        try:
            import sqlalchemy
        except ImportError:
            raise ImportError(
                """To use async Postgres as a memory provider, please install the `sqlalchemy, `psycopg-pool`,
                    `psycopg-binary`, and `psycopg` packages."""
            )

        import controlflow.memory.providers.postgres as postgres_providers

        return postgres_providers.AsyncPostgresMemory()
    raise ValueError(f'Memory provider "{provider}" could not be loaded from a string.')
