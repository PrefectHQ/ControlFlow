# support pipe syntax for unions
from __future__ import annotations

from .settings import settings

from .core.flow import Flow, reset_global_flow as _reset_global_flow, flow
from .core.task import Task, task
from .core.agent import Agent
from .core.controller.controller import Controller
from .instructions import instructions

Flow.model_rebuild()
Task.model_rebuild()
Agent.model_rebuild()

_reset_global_flow()
