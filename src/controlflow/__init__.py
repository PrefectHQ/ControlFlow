from .settings import settings
import controlflow.llm

from .core.agent import Agent
from .core.task import Task
from .core.flow import Flow
from .core.controller.controller import Controller

from .instructions import instructions
from .decorators import flow, task

# --- Default settings ---

from .llm.models import model_from_string, get_default_model
from .core.flow.history import InMemoryHistory, get_default_history

# assign to controlflow.default_model to change the default model
default_model = model_from_string(controlflow.settings.llm_model)
del model_from_string

# assign to controlflow.default_history to change the default history
default_history = InMemoryHistory()
del InMemoryHistory

# assign to controlflow.default_agent to change the default agent
default_agent = Agent(name="Marvin")

# --- Version ---

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "unknown"
