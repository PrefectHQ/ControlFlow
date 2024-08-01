import logging
from typing import TYPE_CHECKING, Optional

from pydantic import Field, field_validator

from controlflow.tools.tools import Tool, tool

from .agent import Agent, AgentResult, BaseAgent

if TYPE_CHECKING:
    from controlflow.orchestration.agent_context import AgentContext

logger = logging.getLogger(__name__)


class Team(BaseAgent):
    """
    A team is a group of agents that can be assigned to a task.
    """

    agents: list[Agent] = Field(description="The agents on the team.")
    name: str = Field("Team of Agents", description="The name of the team.")
    description: Optional[str] = None
    instructions: Optional[str] = Field(
        None,
        description="Instructions for all agents on the team.",
    )
    prompt: Optional[str] = Field(
        None,
        description="A prompt to display as an instruction to any agent selected as part of this team (or a nested team). "
        "Prompts are formatted as jinja templates, with keywords `team: Team` and `context: AgentContext`.",
    )

    _active_agent: Agent = None

    @field_validator("agents", mode="before")
    def validate_agents(cls, v):
        if not v:
            raise ValueError("A team must have at least one agent.")
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._active_agent = self.agents[0]

    def get_agent(self) -> Agent:
        return self._active_agent

    def set_agent(self, agent: Agent):
        self._active_agent = agent

    def serialize_for_prompt(self) -> dict:
        data = self.model_dump(exclude={"agents"})
        data["agents"] = [agent.serialize_for_prompt() for agent in self.agents]
        return data

    def get_prompt(self, context: "AgentContext") -> str:
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.TeamTemplate(
            template=self.prompt, team=self, context=context
        )
        return template.render()

    def get_tools(self) -> list[Tool]:
        return []

    def _run(self, context: "AgentContext"):
        context.add_tools(self.get_tools())
        context.add_prompts([self.get_prompt(context=context)])

        self.pre_run_hook(context=context)
        actions = self.get_agent()._run(context=context)
        self.post_run_hook(context=context, actions=actions)

    async def _run_async(self, context: "AgentContext"):
        context.add_tools(self.get_tools())
        context.add_prompts([self.get_prompt(context=context)])

        self.pre_run_hook(context=context)
        actions = await self.get_agent()._run_async(context=context)
        self.post_run_hook(context=context, actions=actions)

    def pre_run_hook(self, context: "AgentContext"):
        pass

    def post_run_hook(self, context: "AgentContext", actions: "AgentResult"):
        pass


class RoundRobinTeam(Team):
    _agent_index: int = 0

    def post_run_hook(self, context: "AgentContext", actions: "AgentResult") -> Agent:
        if not actions.tool_results:
            self._agent_index = (self._agent_index + 1) % len(self.agents)
            self.set_agent(self.agents[self._agent_index])


class TurnTeam(Team):
    """
    A team that allows each agent to select the next agent to take a turn.
    """

    instructions: str = """Your turn will continue until you use the `end_turn` tool to select another agent."""

    def get_tools(self) -> list[Tool]:
        @tool(
            description="Use this tool to end your turn and let another agent take over. You must supply the ID of an agent on your team."
        )
        def end_turn(agent_id: str):
            agent = next((a for a in self.agents if a.id == agent_id), None)
            if agent is None:
                raise ValueError(
                    f"Agent with id {agent_id} not found in team {self.name}"
                )
            self._active_agent = agent
            return f"Agent {agent.name} has been selected to take the next turn."

        return [end_turn]
