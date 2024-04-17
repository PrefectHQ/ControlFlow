import inspect
import json
from typing import Any, Callable
from uuid import UUID

import prefect
from marvin.types import FunctionTool
from marvin.utilities.asyncio import run_sync
from marvin.utilities.tools import tool_from_function
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.artifacts import ArtifactRequest
from prefect.context import FlowRunContext, TaskRunContext
from pydantic import TypeAdapter

from control_flow.utilities.types import AssistantTool


def create_markdown_artifact(
    key: str,
    markdown: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Markdown artifact.
    """

    tr_context = TaskRunContext.get()
    fr_context = FlowRunContext.get()

    if tr_context:
        task_run_id = task_run_id or tr_context.task_run.id
    if fr_context:
        flow_run_id = flow_run_id or fr_context.flow_run.id

    client = get_prefect_client()
    run_sync(
        client.create_artifact(
            artifact=ArtifactRequest(
                key=key,
                data=markdown,
                description=description,
                type="markdown",
                task_run_id=task_run_id,
                flow_run_id=flow_run_id,
            )
        )
    )


def create_json_artifact(
    key: str,
    data: Any,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a JSON artifact.
    """

    json_data = TypeAdapter(type(data)).dump_json(data, indent=2).decode()

    create_markdown_artifact(
        key=key,
        markdown=f"```json\n{json_data}\n```",
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


def create_python_artifact(
    key: str,
    code: str,
    description: str = None,
    task_run_id: UUID = None,
    flow_run_id: UUID = None,
) -> None:
    """
    Create a Python artifact.
    """

    create_markdown_artifact(
        key=key,
        markdown=f"```python\n{code}\n```",
        description=description,
        task_run_id=task_run_id,
        flow_run_id=flow_run_id,
    )


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


def wrap_prefect_tool(tool: AssistantTool | Callable) -> AssistantTool:
    """
    Wraps a Marvin tool in a prefect task
    """
    if not isinstance(tool, AssistantTool):
        tool = tool_from_function(tool)

    if isinstance(tool, FunctionTool):
        # for functions, we modify the function to become a Prefect task and
        # publish an artifact that contains details about the function call

        if isinstance(tool.function._python_fn, prefect.tasks.Task):
            return tool

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
            passed_args = inspect.signature(original_fn).bind(*args, **kwargs).arguments
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
