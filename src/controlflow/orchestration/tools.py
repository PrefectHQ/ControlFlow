from typing import TYPE_CHECKING, TypeVar

from pydantic import PydanticSchemaGenerationError, TypeAdapter

from controlflow.agents import Agent
from controlflow.events.agent_events import EndTurnEvent
from controlflow.events.task_events import TaskCompleteEvent
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, tool

if TYPE_CHECKING:
    from controlflow.orchestration.controller import Controller

T = TypeVar("T")


def generate_result_schema(result_type: type[T]) -> type[T]:
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


def create_task_success_tool(
    controller: "Controller", task: Task, agent: Agent
) -> Tool:
    """
    Create an agent-compatible tool for marking this task as successful.
    """

    result_schema = generate_result_schema(task.result_type)

    @tool(
        name=f"mark_task_{task.id}_successful",
        description=f"Mark task {task.id} as successful.",
        private=True,
    )
    def succeed(result: result_schema) -> str:  # type: ignore
        result = task.mark_successful(result=result)
        controller.handle_event(TaskCompleteEvent(task=task))
        controller.handle_event(EndTurnEvent(agent=agent))
        return result

    return succeed


def create_task_fail_tool(controller: "Controller", task: Task, agent: Agent) -> Tool:
    """
    Create an agent-compatible tool for failing this task.
    """

    @tool(
        name=f"mark_task_{task.id}_failed",
        description=f"Mark task {task.id} as failed. Only use when technical errors prevent success.",
        private=True,
    )
    def fail(error: str) -> str:
        result = task.mark_failed(message=error)
        controller.handle_event(TaskCompleteEvent(task=task))
        controller.handle_event(EndTurnEvent(agent=agent))
        return result

    return fail


def create_end_turn_tool(controller: "Controller", agent: Agent) -> Tool:
    """
    Create an agent-compatible tool for ending the turn.
    """

    @tool(private=True)
    def end_turn(next_agent_name: str = None) -> str:
        """
        End your turn so another agent can work. You can optionally choose
        the next agent, which can be any other agent assigned to a ready task.
        Choose an agent likely to help you complete your tasks.
        """
        controller.handle_event(
            EndTurnEvent(agent=agent, next_agent_name=next_agent_name)
        )
        return "Turn ended."

    return end_turn
