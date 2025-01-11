# --- Public top-level API ---


from langchain_core.language_models import BaseChatModel
from .settings import settings

from controlflow.defaults import defaults

# base classes
from .agents import Agent
from .tasks import Task
from .flows import Flow

# functions, utilites, and decorators
from .memory import Memory
from .memory.async_memory import AsyncMemory
from .instructions import instructions
from .decorators import flow, task
from .tools import tool
from .run import run, run_async, run_tasks, run_tasks_async, Stream
from .plan import plan
import controlflow.orchestration


# --- Version ---

try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "unknown"
