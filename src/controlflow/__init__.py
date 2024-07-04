# --- Public top-level API ---

from pyexpat import model


from langchain_core.language_models import BaseChatModel
from .settings import settings
from controlflow.defaults import defaults

from .agents import Agent
from .tasks import Task
from .flows import Flow

from .instructions import instructions
from .decorators import flow, task
from .tools import tool
# --- Version ---


try:
    from ._version import version as __version__  # type: ignore
except ImportError:
    __version__ = "unknown"


# def get_default_model() -> BaseChatModel:
#     from controlflow.llm.models import model_from_string

#     if getattr(defaults, "model", None) is None:
#         return model_from_string(settings.llm_model)
#     else:
#         return defaults.model
