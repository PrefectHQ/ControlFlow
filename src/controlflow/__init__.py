# --- Public top-level API ---


from langchain_core.language_models import BaseChatModel
from .settings import settings

from controlflow.defaults import defaults

from .agents import Agent
from .tasks import Task, run, run_async
from .flows import Flow
from .orchestration import turn_strategies


from .instructions import instructions
from .decorators import flow, task
from .tools import tool


# --- Version ---

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "unknown"
