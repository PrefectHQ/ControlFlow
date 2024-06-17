from typing import Optional, TypeVar, Union

from pydantic import Field

from controlflow.core.agent import Agent
from controlflow.core.flow import Flow
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
    parent: Optional[int] = Field(None, description="The parent of this task (if any).")
    agents: list[int] = Field(
        [],
        description="The agents assigned to the task. If empty, the default agent is used.",
    )
    tools: list[int] = Field(
        [],
        description="The tools provided to complete the task. If empty, no tools are provided.",
    )


class Plan(ControlFlowModel):
    objective: str = Field(description="The overall objective of the plan.")
    instructions: Optional[str] = Field(
        None, description="Any optional instructions for carrying out the plan."
    )
    tasks: list[PlanTask] = Field(
        [],
        description="The tasks that make up the plan.",
    )


def plan(
    objective: str,
    instructions: str = None,
    planning_agent: Agent = None,
    agents: list[Agent] = None,
    tools: list[Union[callable, Tool]] = None,
) -> Task:
    """
    Given an objective and instructions for achieving it, generate a plan for
    completing the objective. Each step of the plan will be turned into a task
    objective.
    """
    tools = as_tools(tools or [])

    agent_dict = dict(enumerate(agents or []))
    tool_dict = dict(
        enumerate([t.dict(include={"name", "description"}) for t in tools])
    )

    task = Task(
        objective="""
            Create a plan consisting of multiple tasks to complete the provided objective.
            """,
        instructions="""
            Use your tool to create the plan. Do not post a message or talk out loud.

            Each task should be a discrete, actionable step that contributes to the overall objective.
            
            ## Objective + Instructions
            This is the main objective of your plan. Ultimately it will be turned into a parent task of all the plan tasks you create.
            
            ## Tasks
            - Use `depends_on` to indicate which tasks must be completed before others can start. Tasks can only depend on tasks that come before them in your plan. 
            - Use `parent` to indicate tasks that are subtasks of others.
            
        """,
        context=dict(
            plan_objective=objective,
            plan_instructions=instructions,
            plan_agents=agent_dict,
            plan_tools=tool_dict,
        ),
        agents=[planning_agent] if planning_agent else None,
        result_type=Plan,
    )

    # create a new flow to avoid polluting the main flow's history
    with Flow():
        task.run()

    plan: Plan = task.result

    parent_task = Task(
        objective=plan.objective,
        instructions=plan.instructions,
        agents=[planning_agent] if planning_agent else None,
    )
    task_ids = {}

    for t in plan.tasks:
        task_ids[t.id] = Task(
            objective=t.objective,
            instructions=t.instructions,
            depends_on=[task_ids[i] for i in t.depends_on],
            parent=parent_task if t.parent is None else task_ids[t.parent],
            agents=[agent_dict[i] for i in t.agents] if t.agents else None,
            tools=[tool_dict[i] for i in t.tools],
        )

    return parent_task
