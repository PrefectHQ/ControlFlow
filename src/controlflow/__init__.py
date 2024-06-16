from .settings import settings
import controlflow.llm

# --- Default model ---
# assign to controlflow.default_model to change the default model
from .llm.models import DEFAULT_MODEL as default_model

from .core.flow import Flow
from .core.task import Task
from .core.agent import Agent
from .core.controller.controller import Controller

from .instructions import instructions
from .decorators import flow, task


# --- Default history ---
# assign to controlflow.default_history to change the default history
from .llm.history import DEFAULT_HISTORY as default_history

# --- Default agent ---
# assign to controlflow.default_agent to change the default agent
from .core.agent.agent import DEFAULT_AGENT as default_agent

# --- Version ---
try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "unknown"
