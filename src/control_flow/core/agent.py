import logging
from enum import Enum
from typing import Callable

from marvin.utilities.tools import tool_from_function
from pydantic import Field

from control_flow.utilities.types import Assistant, AssistantTool, ControlFlowModel

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


def talk_to_human(message: str, get_response: bool = True) -> str:
    """
    Send a message to the human user and optionally wait for a response.
    If `get_response` is True, the function will return the user's response,
    otherwise it will return a simple confirmation.
    """
    print(message)
    if get_response:
        response = input("> ")
        return response
    return "Message sent to user"


class Agent(Assistant, ControlFlowModel):
    user_access: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    controller_access: bool = Field(
        False,
        description="If True, the agent will communicate with the controller via messages.",
    )

    def get_tools(self, user_access: bool = None) -> list[AssistantTool | Callable]:
        if user_access is None:
            user_access = self.user_access
        tools = super().get_tools()
        if user_access:
            tools.append(tool_from_function(talk_to_human))
        return tools
