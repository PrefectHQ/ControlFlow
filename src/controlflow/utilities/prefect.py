from contextlib import contextmanager
from typing import (
    Any,
)
from uuid import UUID

import prefect
import prefect.cache_policies
import prefect.serializers
import prefect.tasks
from prefect import get_client as get_prefect_client
from prefect.artifacts import ArtifactRequest
from prefect.context import (
    FlowRunContext,
    TaskRunContext,
)
from pydantic import TypeAdapter

import controlflow


def prefect_task(*args, **kwargs):
    """
    A decorator that creates a Prefect task with ControlFlow defaults
    """

    # TODO: only open in Flow context?

    kwargs.setdefault("log_prints", controlflow.settings.log_prints)
    kwargs.setdefault("cache_policy", prefect.cache_policies.NONE)
    kwargs.setdefault("result_serializer", "json")

    return prefect.task(*args, **kwargs)


def prefect_flow(*args, **kwargs):
    """
    A decorator that creates a Prefect flow with ControlFlow defaults
    """

    kwargs.setdefault("log_prints", controlflow.settings.log_prints)
    kwargs.setdefault("result_serializer", "json")

    return prefect.flow(*args, **kwargs)


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

    client = get_prefect_client(sync_client=True)

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


def prefect_task_context(**kwargs):
    """
    Creates a Prefect task that starts when the context is entered and ends when
    it closes. This is useful for creating a Prefect task that is not tied to a
    specific function but governs a block of code. Note that some features, like
    retries and caching, will not work.
    """
    supported_kwargs = {
        "name",
        "description",
        "task_run_name",
        "tags",
        "version",
        "timeout_seconds",
        "log_prints",
        "on_completion",
        "on_failure",
    }
    unsupported_kwargs = set(kwargs.keys()) - set(supported_kwargs)
    if unsupported_kwargs:
        raise ValueError(
            f"Unsupported keyword arguments for a task context provided: "
            f"{unsupported_kwargs}. Consider using a @task-decorated function instead."
        )

    @contextmanager
    @prefect_task(**kwargs)
    def task_context():
        yield

    return task_context()


def prefect_flow_context(**kwargs):
    """
    Creates a Prefect flow that starts when the context is entered and ends when
    it closes. This is useful for creating a Prefect flow that is not tied to a
    specific function but governs a block of code. Note that some features, like
    retries and caching, will not work.
    """

    supported_kwargs = {
        "name",
        "description",
        "flow_run_name",
        "tags",
        "version",
        "timeout_seconds",
        "log_prints",
        "on_completion",
        "on_failure",
    }
    unsupported_kwargs = set(kwargs.keys()) - set(supported_kwargs)
    if unsupported_kwargs:
        raise ValueError(
            f"Unsupported keyword arguments for a flow context provided: "
            f"{unsupported_kwargs}. Consider using a @flow-decorated function instead."
        )

    @contextmanager
    @prefect_flow(**kwargs)
    def flow_context():
        yield

    return flow_context()
