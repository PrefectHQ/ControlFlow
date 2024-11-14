from dataclasses import Field
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional

from pydantic.functional_serializers import PlainSerializer

from controlflow.agents.agent import Agent
from controlflow.events.base import Event, UnpersistedEvent

if TYPE_CHECKING:
    from controlflow.orchestration.conditions import RunContext
    from controlflow.orchestration.orchestrator import Orchestrator


# Orchestrator events
class OrchestratorStart(UnpersistedEvent):
    event: Literal["orchestrator-start"] = "orchestrator-start"
    persist: bool = False
    orchestrator: "Orchestrator"
    run_context: "RunContext"


class OrchestratorEnd(UnpersistedEvent):
    event: Literal["orchestrator-end"] = "orchestrator-end"
    persist: bool = False
    orchestrator: "Orchestrator"
    run_context: "RunContext"


class OrchestratorError(UnpersistedEvent):
    event: Literal["orchestrator-error"] = "orchestrator-error"
    persist: bool = False
    orchestrator: "Orchestrator"
    error: Annotated[Exception, PlainSerializer(lambda x: str(x), return_type=str)]


class AgentTurnStart(UnpersistedEvent):
    event: Literal["agent-turn-start"] = "agent-turn-start"
    persist: bool = False
    orchestrator: "Orchestrator"
    agent: Agent


class AgentTurnEnd(UnpersistedEvent):
    event: Literal["agent-turn-end"] = "agent-turn-end"
    persist: bool = False
    orchestrator: "Orchestrator"
    agent: Agent
