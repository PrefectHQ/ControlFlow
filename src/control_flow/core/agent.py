import logging
from typing import Callable

from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from marvin.utilities.tools import tool_from_function
from pydantic import Field

from control_flow.core.flow import get_flow
from control_flow.core.task import Task
from control_flow.utilities.prefect import (
    wrap_prefect_tool,
)
from control_flow.utilities.types import Assistant, AssistantTool, ControlFlowModel
from control_flow.utilities.user_access import talk_to_human

logger = logging.getLogger(__name__)


class Agent(Assistant, ControlFlowModel, ExposeSyncMethodsMixin):
    name: str = "Agent"
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )

    def get_tools(self) -> list[AssistantTool | Callable]:
        tools = super().get_tools()
        if self.user_access:
            tools.append(tool_from_function(talk_to_human))

        return [wrap_prefect_tool(tool) for tool in tools]

    @expose_sync_method("run")
    async def run_async(self, tasks: list[Task] | Task | None = None):
        from control_flow.core.controller import Controller

        if isinstance(tasks, Task):
            tasks = [tasks]

        controller = Controller(agents=[self], tasks=tasks or [], flow=get_flow())
        return await controller.run_agent_async(agent=self)

    def __hash__(self):
        return id(self)
