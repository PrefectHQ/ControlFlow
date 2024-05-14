from .settings import settings

# from .agent_old import task, Agent, run_ai
from .core.flow import Flow, reset_global_flow as _reset_global_flow, flow
from .core.task import Task, task
from .core.agent import Agent
from .core.controller.controller import Controller
from .instructions import instructions
from .dx import run_ai

Flow.model_rebuild()
Task.model_rebuild()
Agent.model_rebuild()

_reset_global_flow()
