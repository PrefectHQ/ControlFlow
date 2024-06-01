import inspect
from typing import Any
from uuid import UUID

from prefect import get_client as get_prefect_client
from prefect.artifacts import ArtifactRequest
from prefect.context import FlowRunContext, TaskRunContext
from prefect.utilities.asyncutils import run_coro_as_sync
from pydantic import TypeAdapter


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
    run_coro_as_sync(
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

    try:
        markdown = TypeAdapter(type(data)).dump_json(data, indent=2).decode()
        markdown = f"```json\n{markdown}\n```"
    except Exception:
        markdown = str(data)

    create_markdown_artifact(
        key=key,
        markdown=markdown,
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


# def wrap_prefect_tool(tool: ToolType) -> AssistantTool:
#     if not (isinstance(tool, AssistantTool) or isinstance(tool, ToolFunction)):
#         tool = tool(tool)

#     if isinstance(tool, ToolFunction):
#         # for functions, we modify the function to become a Prefect task and
#         # publish an artifact that contains details about the function call

#         if isinstance(tool.function._python_fn, prefect.tasks.Task):
#             return tool

#         def modified_fn(
#             # provide default args to avoid a late-binding issue
#             original_fn: Callable = tool.function._python_fn,
#             tool: ToolFunction = tool,
#             **kwargs,
#         ):
#             # call fn
#             result = original_fn(**kwargs)

#             # prepare artifact
#             passed_args = inspect.signature(original_fn).bind(**kwargs).arguments
#             try:
#                 passed_args = json.dumps(passed_args, indent=2)
#             except Exception:
#                 pass
#             create_markdown_artifact(
#                 markdown=TOOL_CALL_FUNCTION_RESULT_TEMPLATE.format(
#                     name=tool.function.name,
#                     description=tool.function.description or "(none provided)",
#                     args=passed_args,
#                     result=result,
#                 ),
#                 key="result",
#             )

#             # return result
#             return result

#         # replace the function with the modified version
#         tool.function._python_fn = prefect_task(
#             modified_fn,
#             task_run_name=f"Tool call: {tool.function.name}",
#         )

#     return tool
