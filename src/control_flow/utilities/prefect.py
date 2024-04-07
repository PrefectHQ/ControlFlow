from typing import Any
from uuid import UUID

from marvin.utilities.asyncio import run_sync
from prefect import get_client as get_prefect_client
from prefect.artifacts import ArtifactRequest
from prefect.context import FlowRunContext, TaskRunContext
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
