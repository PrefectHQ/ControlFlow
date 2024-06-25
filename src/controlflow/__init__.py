from .settings import settings
from . import llm

from .agents import Agent
from .tasks import Task
from .flows import Flow
from .controllers import Controller

from .instructions import instructions
from .decorators import flow, task
from .llm.tools import tool

# --- Default settings ---

from .llm.models import model_from_string, get_default_model
from .flows.history import InMemoryHistory, get_default_history

# assign to controlflow.default_model to change the default model
default_model = model_from_string(settings.llm_model)
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
