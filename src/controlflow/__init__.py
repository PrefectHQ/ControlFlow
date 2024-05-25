from .settings import settings

from . import llm

from .core.flow import Flow
from .core.task import Task
from .core.agent import Agent
from .core.controller.controller import Controller

from .instructions import instructions
from .decorators import flow, task

# --- Default agent ---

from .core.agent import DEFAULT_AGENT

default_agent = DEFAULT_AGENT
del DEFAULT_AGENT

# --- Default history ---

from .llm.history import DEFAULT_HISTORY

default_history = DEFAULT_HISTORY
del DEFAULT_HISTORY
