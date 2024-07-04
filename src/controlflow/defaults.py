from typing import Any, Optional

from pydantic import field_validator

import controlflow
import controlflow.utilities
import controlflow.utilities.logging
from controlflow.llm.models import BaseChatModel
from controlflow.utilities.types import ControlFlowModel

from .agents import Agent
from .events.history import History, InMemoryHistory
from .llm.models import _get_initial_default_model, model_from_string

__all__ = ["defaults"]

logger = controlflow.utilities.logging.get_logger(__name__)


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
    # add more defaults here

    def __repr__(self) -> str:
        fields = ", ".join(self.model_fields.keys())
        return f"<ControlFlow Defaults: {fields}>"

    @field_validator("model")
    def _model(cls, v):
        if isinstance(v, str):
            v = model_from_string(v)
        elif v is not None and not isinstance(v, BaseChatModel):
            raise ValueError("Input must be an instance of BaseChatModel")
        return v


defaults = Defaults(
    model=_get_initial_default_model(),
    history=InMemoryHistory(),
    agent=Agent(name="Marvin"),
)
