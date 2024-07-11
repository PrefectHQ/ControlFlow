from typing import Literal

from controlflow.events.base import UnpersistedEvent
from controlflow.orchestration.orchestrator import Orchestrator


class OrchestratorStart(UnpersistedEvent):
    event: Literal["orchestrator-start"] = "orchestrator-start"
    persist: bool = False
    orchestrator: Orchestrator


class OrchestratorEnd(UnpersistedEvent):
    event: Literal["orchestrator-end"] = "orchestrator-end"
    persist: bool = False
    orchestrator: Orchestrator


class OrchestratorError(UnpersistedEvent):
    event: Literal["orchestrator-error"] = "orchestrator-error"
    persist: bool = False
    orchestrator: Orchestrator
    error: Exception
