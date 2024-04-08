import logging
from typing import Any

from marvin.beta.assistants import PrintHandler, Run
from marvin.utilities.asyncio import ExposeSyncMethodsMixin
from pydantic import BaseModel, Field, field_validator, model_validator

from control_flow.core.agent import Agent, AgentStatus
from control_flow.core.controller.delegation import (
    DelegationStrategy,
    RoundRobin,
    Single,
)
from control_flow.core.flow import Flow
from control_flow.instructions import get_instructions
from control_flow.utilities.context import ctx
from control_flow.utilities.types import Thread

logger = logging.getLogger(__name__)


class Controller(BaseModel, ExposeSyncMethodsMixin):
    agents: list[Agent]
    flow: Flow = Field(None, validate_default=True)
    delegation_strategy: DelegationStrategy = Field(
        validate_default=True,
        description="The strategy for delegating work to assistants.",
        default_factory=RoundRobin,
    )
    # termination_strategy: TerminationStrategy
    context: dict = {}
    instructions: str = None
    model_config: dict = dict(extra="forbid")

    @field_validator("flow", mode="before")
    def _default_flow(cls, v):
        if v is None:
            v = ctx.get("flow", None)
        return v

    @field_validator("agents", mode="before")
    def _validate_agents(cls, v):
        if not v:
            raise ValueError("At least one agent is required.")
        return v

    async def run(self):
        """
        Run the control flow.
        """

        while incomplete_agents := [
            a for a in self.agents if a.status == AgentStatus.INCOMPLETE
        ]:
            agent = self.delegation_strategy(incomplete_agents)
            if not agent:
                return
            await self.run_agent(agent=agent)

    async def run_agent(self, agent: Agent, thread: Thread = None):
        """
        Run a single agent.
        """
        from control_flow.core.controller.instruction_template import MainTemplate

        instructions_template = MainTemplate(
            agent=agent,
            controller=self,
            context=self.context,
            instructions=get_instructions(),
        )

        run = Run(
            assistant=agent.assistant,
            thread=thread or self.flow.thread,
            instructions=instructions_template.render(),
            tools=agent.get_tools() + self.flow.tools,
            event_handler_class=PrintHandler,
        )

        await run.run_async()

        return run


class SingleAgentController(Controller):
    """
    A SingleAgentController is a controller that runs a single agent.
    """

    delegation_strategy: Single = Field(None, validate_default=True)

    @field_validator("agents", mode="before")
    def _validate_agents(cls, v):
        if len(v) != 1:
            raise ValueError("A SingleAgentController must have exactly one agent.")
        return v

    @model_validator(mode="before")
    @classmethod
    def _create_single_strategy(cls, data: Any) -> Any:
        """
        Create a Single delegation strategy with the agent.
        """
        data["delegation_strategy"] = Single(agent=data["agents"][0])
        return data
