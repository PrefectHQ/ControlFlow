# --- Public top-level API ---

from .settings import settings

from .agents import Agent
from .tasks import Task
from .flows import Flow

from .instructions import instructions
from .decorators import flow, task
from .llm.tools import tool

# --- Default settings ---

from .llm.models import _get_initial_default_model, get_default_model
from .flows.history import InMemoryHistory, get_default_history

# assign to controlflow.default_model to change the default model
default_model = _get_initial_default_model()
del _get_initial_default_model

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
