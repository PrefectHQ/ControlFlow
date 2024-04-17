import logging
from typing import Self

from marvin.beta.assistants import PrintHandler, Run
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from pydantic import BaseModel, Field, field_validator, model_validator

from control_flow.core.agent import Agent
from control_flow.core.controller.delegation import (
    DelegationStrategy,
    RoundRobin,
)
from control_flow.core.flow import Flow
from control_flow.core.task import Task, TaskStatus
from control_flow.instructions import get_instructions as get_context_instructions
from control_flow.utilities.types import Thread

logger = logging.getLogger(__name__)


class Controller(BaseModel, ExposeSyncMethodsMixin):
    flow: Flow
    agents: list[Agent]
    tasks: list[Task] = Field(
        description="Tasks that the controller will complete.",
        default_factory=list,
    )
    delegation_strategy: DelegationStrategy = Field(
        validate_default=True,
        description="The strategy for delegating work to assistants.",
        default_factory=RoundRobin,
    )
    # termination_strategy: TerminationStrategy
    context: dict = {}
    instructions: str = None
    user_access: bool | None = Field(
        None,
        description="If True or False, overrides the user_access of the "
        "agents. If None, the user_access setting of each agents is used.",
    )
    model_config: dict = dict(extra="forbid")

    @field_validator("agents", mode="before")
    def _validate_agents(cls, v):
        if not v:
            raise ValueError("At least one agent is required.")
        return v

    @model_validator(mode="after")
    def _add_tasks_to_flow(self) -> Self:
        for task in self.tasks:
            self.flow.add_task(task)
        return self

    @expose_sync_method("run")
    async def run_async(self):
        """
        Run the control flow.
        """

        while True:
            incomplete = any([t for t in self.tasks if t.status == TaskStatus.PENDING])
            if not incomplete:
                break
            if len(self.agents) > 1:
                agent = self.delegation_strategy(self.agents)
            else:
                agent = self.agents[0]
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
            instructions=get_context_instructions(),
        )

        instructions = instructions_template.render()

        tools = self.flow.tools + agent.get_tools(user_access=self.user_access)

        for task in self.tasks:
            task_id = self.flow.get_task_id(task)
            tools = tools + task.get_tools(task_id=task_id)

        run = Run(
            assistant=agent,
            thread=thread or self.flow.thread,
            instructions=instructions,
            tools=tools,
            event_handler_class=PrintHandler,
        )

        await run.run_async()

        return run

    def task_ids(self) -> dict[Task, int]:
        return {task: self.flow.get_task_id(task) for task in self.tasks}
