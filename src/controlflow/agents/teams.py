import abc
import logging
from typing import TYPE_CHECKING, Optional

from pydantic import Field, field_validator

from .agent import Agent, BaseAgent

if TYPE_CHECKING:
    from controlflow.orchestration.agent_context import AgentContext

logger = logging.getLogger(__name__)


class BaseTeam(BaseAgent):
    """
    A team is a group of agents that can be assigned to a task.

    Each team consists of one or more agents, and the only requirement for a
    team is to implement the `get_agent` method. This method should return one of
    the agents in the team, based on some logic that determines which agent should go next.
    """

    name: str = Field("Team", description="The name of the team.")
    instructions: Optional[str] = Field(
        None,
        description="Instructions for all agents on the team, private to this agent.",
    )
    prompt: Optional[str] = Field(
        None,
        description="A prompt to display as an instruction to any agent selected as part of this team (or a nested team). "
        "Prompts are formatted as jinja templates, with keywords `team: Team` and `context: AgentContext`.",
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

    def get_prompt(self, context: "AgentContext") -> str:
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.TeamTemplate(
            template=self.prompt, team=self, context=context
        )
        return template.render()

    def _run(self, context: "AgentContext"):
        context.add_agent(self)
        context.add_instructions([self.get_prompt(context=context)])
        agent = self.get_agent(context=context)
        agent._run(context=context)
        self._iterations += 1

    async def _run_async(self, context: "AgentContext"):
        context.add_agent(self)
        context.add_instructions([self.get_prompt(context=context)])
        agent = self.get_agent(context=context)
        await agent._run_async(context=context)
        self._iterations += 1


class Team(BaseTeam):
    """
    The most basic team operates in a round robin fashion
    """

    def get_agent(self, context: "AgentContext"):
        # if the last event was a tool result, it should be shown to the same agent instead of advancing to the next agent
        last_agent_event = context.flow.get_events(
            agents=self.agents,
            tasks=context.tasks,
            types=["tool-result", "agent-message"],
            limit=1,
        )
        if (
            last_agent_event
            and last_agent_event[0].event == "tool-result"
            and not last_agent_event[0].tool_result.end_turn
        ):
            return last_agent_event[0].agent

        return self.agents[self._iterations % len(self.agents)]
