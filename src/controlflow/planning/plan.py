from typing import Optional, TypeVar, Union

from pydantic import Field

from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.tasks.task import Task
from controlflow.tools import Tool, as_tools
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


def create_plan(
    objective: str,
    instructions: str = None,
    planning_agent: Agent = None,
    agents: list[Agent] = None,
    tools: list[Union[callable, Tool]] = None,
    context: dict = None,
) -> list[Task]:
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
            Use your mark_successful tool to create the plan. Do not post a
            message or talk out loud.

            Each task should be a discrete, actionable step that contributes to
            the overall objective. Do not waste time on uneccessary or redundant
            steps. Make sure to use your tools. 
            
            When creating tasks, imagine that you had to complete the plan
            yourself. What steps would you take? What tools would you use? What
            information would you need? Remember that each task has a token cost
            (both in its evaluation and needing to mark it complete), so try to
            organize objectives by outcomes and dependencies, not by the actions
            you'd need to take.
            
            - Use `depends_on` to indicate which tasks must be completed before
              others can start. Tasks can only depend on tasks that come before
              them in your plan. 
            - Use `parent` to indicate tasks that are subtasks of others.
            - Don't create needless tasks like "document the findings"
            
        """,
        context=dict(
            plan_objective=objective,
            plan_instructions=instructions,
            plan_agents=agent_dict,
            plan_tools=tool_dict,
        )
        | context
        or {},
        agents=[planning_agent] if planning_agent else None,
        result_type=list[PlanTask],
        infer_parent=False,
    )

    # create a new flow to avoid polluting the main flow's history
    with Flow():
        task.run()

    plan: list[PlanTask] = task.result

    task_ids = {}

    for t in plan:
        try:
            task_agents = [agent_dict[i] for i in t.agents] if t.agents else None
        except KeyError:
            task_agents = None

        try:
            task_tools = [tool_dict[i] for i in t.tools]
        except KeyError:
            task_tools = []

        task_ids[t.id] = Task(
            objective=t.objective,
            instructions=t.instructions,
            depends_on=[task_ids[i] for i in t.depends_on],
            parent=task_ids[t.parent] if t.parent else None,
            agents=task_agents,
            tools=task_tools,
            context=context,
        )

    return list(task_ids.values())
