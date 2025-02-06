from typing import Any, Optional, Union

from pydantic import field_validator

import controlflow
import controlflow.utilities
import controlflow.utilities.logging
from controlflow.llm.models import BaseChatModel
from controlflow.memory.async_memory import AsyncMemoryProvider, get_memory_provider
from controlflow.memory.memory import MemoryProvider, get_memory_provider
from controlflow.utilities.general import ControlFlowModel

from .agents import Agent
from .events.history import FileHistory, History, InMemoryHistory
from .llm.models import _get_initial_default_model, get_model

__all__ = ["defaults"]

logger = controlflow.utilities.logging.get_logger(__name__)

_default_model = _get_initial_default_model()
_default_history = InMemoryHistory()
_default_agent = Agent(name="Marvin")
try:
    _default_memory_provider = get_memory_provider(controlflow.settings.memory_provider)
except Exception:
    _default_memory_provider = controlflow.settings.memory_provider


class Defaults(ControlFlowModel):
    """
    This class holds the default values for various parts of the ControlFlow
    library.

    Note that users should interact with the `defaults` object directly, rather
    than instantiating this class. It is intended to be created once, when ControlFlow
    is imported, and then used as a singleton.
    """

    model: Optional[Any]
    history: History
    agent: Agent
    memory_provider: Optional[Union[MemoryProvider, AsyncMemoryProvider, str]]

    # add more defaults here
    def __repr__(self) -> str:
        fields = ", ".join(self.model_fields.keys())
        return f"<ControlFlow Defaults: {fields}>"

    @field_validator("model", mode="before")
    def _model(cls, v):
        if isinstance(v, str):
            v = get_model(v)
        # the model validator in langchain forcibly expects a dictionary
        elif v is not None and not isinstance(v, (dict, BaseChatModel)):
            raise ValueError("Input must be an instance of dict or BaseChatModel")
        return v


defaults = Defaults(
    model=_default_model,
    history=_default_history,
    agent=_default_agent,
    memory_provider=_default_memory_provider,
)
