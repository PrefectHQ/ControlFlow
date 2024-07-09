import abc
import logging
import random
from typing import TYPE_CHECKING, Optional

from pydantic import Field, field_validator

from controlflow.agents.names import TEAMS

from .agent import Agent, BaseAgent

if TYPE_CHECKING:
    from controlflow.orchestration.agent_context import AgentContext

logger = logging.getLogger(__name__)


class Team(BaseAgent):
    name: str = Field(
        description="The name of the team.",
        default_factory=lambda: random.choice(TEAMS),
    )
    instructions: Optional[str] = Field(
        None,
        description="Instructions for all agents on the team, private to this agent.",
    )

    agents: list[Agent] = Field(
        description="The agents in the team.",
        default_factory=list,
    )
    _iterations: int = 0

    @field_validator("agents", mode="before")
    def validate_agents(cls, v):
        if not v:
            raise ValueError("A team must have at least one agent.")
        return v

    def serialize_for_prompt(self) -> dict:
        data = self.model_dump(exclude={"agents"})
        data["agents"] = [agent.serialize_for_prompt() for agent in self.agents]
        return data

    @abc.abstractmethod
    def get_agent(self, context: "AgentContext") -> Agent:
        raise NotImplementedError()

    def get_prompt(self) -> str:
        from controlflow.orchestration.prompts import TeamTemplate

        return TeamTemplate(team=self).render()

    def _run(self, context: "AgentContext"):
        agent = self.get_agent(context=context)
        context.prompts.team = self.get_prompt()
        with context.with_agent(agent) as agent_context:
            agent._run(context=agent_context)
        self._iterations += 1

    async def _run_async(self, context: "AgentContext"):
        agent = self.get_agent(context=context)
        await agent._run_async(context=context)
        self._iterations += 1


class RoundRobinTeam(Team):
    def get_agent(self, context: "AgentContext"):
        # TODO: only advance agent if a tool wasn't used
        return self.agents[self._iterations % len(self.agents)]
