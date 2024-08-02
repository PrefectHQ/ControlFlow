from typing import Any, Callable, Optional, Union

from pydantic import Field, field_validator

import controlflow
import controlflow.utilities
import controlflow.utilities.logging
from controlflow.llm.models import BaseChatModel
from controlflow.utilities.general import ControlFlowModel

from .agents import Agent, Team
from .events.history import History, InMemoryHistory
from .llm.models import _get_initial_default_model, model_from_string

__all__ = ["defaults"]

logger = controlflow.utilities.logging.get_logger(__name__)

_default_model = _get_initial_default_model()
_default_history = InMemoryHistory()
_default_agent = Agent(name="Marvin")
_default_team = Team


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
    team: Union[type[Team], Callable[[list[Agent]], Team]] = Field(
        description="A class or callable that accepts a list of Agents and returns a Team."
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "team":
            logger.warning(
                "The default team interface is not final and may change in the future."
            )
        return super().__setattr__(name, value)

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
    model=_default_model,
    history=_default_history,
    agent=_default_agent,
    team=_default_team,
)
