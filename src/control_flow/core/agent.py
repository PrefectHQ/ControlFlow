import logging
from enum import Enum
from typing import Callable

from marvin.utilities.tools import tool_from_function
from pydantic import Field

from control_flow.utilities.prefect import (
    wrap_prefect_tool,
)
from control_flow.utilities.types import Assistant, AssistantTool, ControlFlowModel
from control_flow.utilities.user_access import talk_to_human

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class Agent(Assistant, ControlFlowModel):
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    controller_access: bool = Field(
        False,
        description="If True, the agent will communicate with the controller via messages.",
    )

    def get_tools(self) -> list[AssistantTool | Callable]:
        tools = super().get_tools()
        if self.user_access:
            tools.append(tool_from_function(talk_to_human))

        return [wrap_prefect_tool(tool) for tool in tools]
