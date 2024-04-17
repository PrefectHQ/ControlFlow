import inspect
import json
import logging
from enum import Enum
from typing import Callable

from marvin.types import FunctionTool
from marvin.utilities.tools import tool_from_function
from prefect import task as prefect_task
from pydantic import Field

from control_flow.utilities.prefect import (
    create_markdown_artifact,
)
from control_flow.utilities.types import Assistant, AssistantTool, ControlFlowModel

logger = logging.getLogger(__name__)

TOOL_CALL_FUNCTION_RESULT_TEMPLATE = inspect.cleandoc(
    """
    ## Tool call: {name}
    
    **Description:** {description}
    
    ## Arguments
    
    ```json
    {args}
    ```
    
    ### Result
    
    ```json
    {result}
    ```
    """
)


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

    def get_tools(self) -> list[AssistantTool | Callable]:
        tools = super().get_tools()
        if self.user_access:
            tools.append(tool_from_function(talk_to_human))

        wrapped_tools = []
        for tool in tools:
            wrapped_tools.append(self._wrap_prefect_tool(tool))
        return tools

    def _wrap_prefect_tool(self, tool: AssistantTool | Callable) -> AssistantTool:
        if not isinstance(tool, AssistantTool):
            tool = tool_from_function(tool)

        if isinstance(tool, FunctionTool):
            # for functions, we modify the function to become a Prefect task and
            # publish an artifact that contains details about the function call

            async def modified_fn(
                *args,
                # provide default args to avoid a late-binding issue
                original_fn: Callable = tool.function._python_fn,
                tool: FunctionTool = tool,
                **kwargs,
            ):
                # call fn
                result = original_fn(*args, **kwargs)

                # prepare artifact
                passed_args = (
                    inspect.signature(original_fn).bind(*args, **kwargs).arguments
                )
                try:
                    passed_args = json.dumps(passed_args, indent=2)
                except Exception:
                    pass
                create_markdown_artifact(
                    markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
                        name=tool.function.name,
                        description=tool.function.description or "(none provided)",
                        args=passed_args,
                        result=result,
                    ),
                    key="result",
                )

                # return result
                return result

            # replace the function with the modified version
            tool.function._python_fn = prefect_task(
                modified_fn,
                task_run_name=f"Tool call: {tool.function.name}",
            )

        return tool
