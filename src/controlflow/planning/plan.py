from typing import Optional, TypeVar, Union

from pydantic import Field

from controlflow.core.agent import Agent
from controlflow.core.task import Task
from controlflow.llm.tools import Tool, as_tools
from controlflow.utilities.types import ControlFlowModel

ToolLiteral = TypeVar("ToolLiteral", bound=str)


class PlanTask(ControlFlowModel):
    id: int
    objective: str
    instructions: Optional[str] = Field(
        None,
        description="Any additional instructions for completing the task objective.",
    )
    depends_on: list[int] = Field(
        [], description="Tasks that must be completed before this task can be started."
    )


class Plan(ControlFlowModel):
    tasks: list[PlanTask]


def plan(
    objective: str,
    instructions: str = None,
    agent: Agent = None,
    tools: list[Union[callable, Tool]] = None,
) -> Task:
    """
    Given an objective and instructions for achieving it, generate a plan for
    completing the objective. Each step of the plan will be turned into a task
    objective.
    """
    agents = [agent] if agent else None
    plan_task = Task(
        objective="""
            Create a plan to complete the provided objective. Each step of your plan
            will be turned into a task objective, like this one. After generating
            the plan, you will be tasked with executing it, using your tools and any additional ones provided.
            """,
        instructions="""
        Indicate dependencies between tasks, including sequential dependencies.
        """,
        result_type=Plan,
        context=dict(
            plan_objective=objective,
            plan_instructions=instructions,
            plan_tools=[
                t.dict(include={"name", "description"}) for t in as_tools(tools or [])
            ],
        ),
        agents=agents,
    )

    plan_task.run()

    parent_task = Task(objective=objective, agents=agents)
    task_ids = {}

    subtask: PlanTask
    for subtask in plan_task.result.tasks:
        task_ids[subtask.id] = Task(
            objective=subtask.objective,
            instructions=subtask.instructions,
            parent=parent_task,
            depends_on=[task_ids[task_id] for task_id in subtask.depends_on],
            agents=agents,
            tools=tools,
        )

    return parent_task
