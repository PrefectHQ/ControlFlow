from typing import TypeVar

from pydantic import PydanticSchemaGenerationError, TypeAdapter

from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, tool

T = TypeVar("T")


def _generate_result_schema(result_type: type[T]) -> type[T]:
    if result_type is None:
        return None

    result_schema = None
    # try loading pydantic-compatible schemas
    try:
        TypeAdapter(result_type)
        result_schema = result_type
    except PydanticSchemaGenerationError:
        pass
    # try loading as dataframe
    # try:
    #     import pandas as pd

    #     if result_type is pd.DataFrame:
    #         result_schema = PandasDataFrame
    #     elif result_type is pd.Series:
    #         result_schema = PandasSeries
    # except ImportError:
    #     pass
    if result_schema is None:
        raise ValueError(
            f"Could not load or infer schema for result type {result_type}. "
            "Please use a custom type or add compatibility."
        )
    return result_schema


def create_task_success_tool(task: Task) -> Tool:
    """
    Create an agent-compatible tool for marking this task as successful.
    """

    result_schema = _generate_result_schema(task.result_type)

    @tool(
        name=f"mark_task_{task.id}_successful",
        description=f"Mark task {task.id} as successful.",
        private=True,
        include_return_description=False,
    )
    def succeed(result: result_schema) -> str:  # type: ignore
        if task.is_successful():
            raise ValueError(f"{task.friendly_name()} is already marked successful.")
        task.mark_successful(result=result)
        return f"{task.friendly_name()} marked successful."

    return succeed


def create_task_fail_tool(task: Task) -> Tool:
    """
    Create an agent-compatible tool for failing this task.
    """

    @tool(
        name=f"mark_task_{task.id}_failed",
        description=(
            f"Mark task {task.id} as failed. Only use when technical errors prevent success. Provide a detailed reason for the failure."
        ),
        private=True,
        include_return_description=False,
    )
    def fail(reason: str) -> str:
        task.mark_failed(reason=reason)
        return f"{task.friendly_name()} marked failed."

    return fail
