import abc
import uuid
from typing import ClassVar, Optional, cast

from pydantic import Field

from controlflow.tools import Tool
from controlflow.utilities.context import ctx
from controlflow.utilities.types import ControlFlowModel


class Memory(ControlFlowModel, abc.ABC):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    def load(self) -> dict[int, str]:
        """
        Load all memories as a dictionary of index to value.
        """
        raise NotImplementedError()

    def update(self, value: str, index: int = None):
        """
        Store a value, optionally overwriting an existing value at the given index.
        """
        raise NotImplementedError()

    def delete(self, index: int):
        raise NotImplementedError()

    def get_tools(self) -> list[Tool]:
        update_tool = Tool.from_function(
            self.update,
            name="update_memory",
            description="Privately remember an idea or fact, optionally updating the existing memory at `index`",
        )
        delete_tool = Tool.from_function(
            self.delete,
            name="delete_memory",
            description="Forget the private memory at `index`",
        )

        tools = [update_tool, delete_tool]

        return tools


class AgentMemory(Memory):
    """
    In-memory store for an agent. Memories are scoped to the agent.

    Note memories may persist across flows.
    """

    _memory: list[str] = []

    def update(self, value: str, index: int = None):
        if index is not None:
            self._memory[index] = value
        else:
            self._memory.append(value)

    def load(self, thread_id: str) -> dict[int, str]:
        return dict(enumerate(self._memory))

    def delete(self, index: int):
        del self._memory[index]


class ThreadMemory(Memory):
    """
    In-memory store for an agent. Memories are scoped to each thread.
    """

    _memory: ClassVar[dict[str, list[str]]] = {}

    def _get_thread_id(self) -> Optional[str]:
        from controlflow.flows import Flow

        if flow := ctx.get("flow", None):  # type: Flow
            flow = cast(Flow, flow)
            return flow.thread_id

    def update(self, value: str, index: int = None):
        thread_id = self._get_thread_id()
        if index is not None:
            self._memory[thread_id][index] = value
        else:
            self._memory[thread_id].append(value)

    def load(self, thread_id: str) -> dict[int, str]:
        return dict(enumerate(self._memory.get(thread_id, [])))

    def delete(self, index: int):
        thread_id = self._get_thread_id()
        del self._memory[thread_id][index]
