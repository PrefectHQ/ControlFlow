from typing import Optional, TypeVar, Union

from pydantic import Field

import controlflow
from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.tasks.task import Task
from controlflow.tools import Tool, as_tools
from controlflow.utilities.general import ControlFlowModel

ToolLiteral = TypeVar("ToolLiteral", bound=str)


class PlanTask(ControlFlowModel):
    id: int
    objective: str = Field(
        description="The objective of the task. This should be a concise statement of the task's purpose.",
    )
    instructions: Optional[str] = Field(
        None,
        description="Any additional instructions for completing the task objective.",
    )
    depends_on: list[int] = Field(
        [],
        description="Tasks that must be completed before this task can be started. Must be the id of one of the other tasks in the plan.",
    )
    parent: Optional[int] = Field(
        None,
        description="The parent of this task (if any). Must be the id of one of the other tasks in the plan.",
    )
    agents: list[int] = Field(
        description="The agents assigned to the task. Must be the index of one of the agents provided in the plan_agents context variable.",
    )
    tools: list[int] = Field(
        [],
        description="The tools provided to complete the task, if any. Must be the index of one of the tools provided in the plan_tools context variable.",
    )


def plan(
    objective: str,
    instructions: Optional[str] = None,
    agent: Optional[Agent] = None,
    agents: Optional[list[Agent]] = None,
    tools: list[Union[callable, Tool]] = None,
    context: Optional[dict] = None,
    n_tasks: Optional[int] = None,
) -> list[Task]:
    """
    Given an objective and instructions for achieving it, generate a plan for
    completing the objective. Each step of the plan will be turned into a task
    objective.
    """
    tools = as_tools(tools or [])

    if agent is None:
        agent = controlflow.defaults.agent
    if not agents:
        agents = [agent]

    agent_dict = dict(enumerate(agents))
    tool_dict = dict(
        enumerate([t.model_dump(include={"name", "description"}) for t in tools])
    )

    def validate_plan(plan: list[PlanTask]):
        if n_tasks and len(plan) != n_tasks:
            raise ValueError(f"Expected {n_tasks} tasks, got {len(plan)}")
        for task in plan:
            if any(a not in agent_dict for a in task.agents):
                raise ValueError(
                    f"Not all agents in task {task.id} are valid: {task.agents}"
                )
            if any(t not in tool_dict for t in task.tools):
                raise ValueError(
                    f"Not all tools in task {task.id} are valid: {task.tools}"
                )
        return plan

    plan_task = Task(
        objective="""
            Create a plan consisting of ControlFlow tasks that will allow agents
            to achieve the provided objective.
            """,
        instructions="""
            Use your mark_successful tool to create the plan. Do not post a
            message or talk out loud.
            
            If specified, the task must include exactly `number_of_tasks` tasks;
            otherwise, follow your judgement or any additional instructions.

            Each task should be a discrete, actionable step that contributes to
            the overall objective. Do not waste time on uneccessary or redundant
            steps. Take tools and agent capabilities into account when creating
            a task. Do not create unachievable tasks, like "search for X" if
            your agent or tools do not have a search capability. You may,
            however, create tasks that serve as discrete or useful checkpoints
            for completing the overall objective. Do not create tasks for
            "verifying" results unless you have agents or tools to deploy that
            will truly lead to a differentiated outcome.
            
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
            - Assign agents and tools to tasks to help manage the plan. Try not
              to assign agents unless they are needed.
            - Don't create needless tasks like "document the findings." Only
              create tasks whose results are useful checkpoints for completing
              the overall objective.
            
        """,
        context=dict(
            plan_objective=objective,
            plan_instructions=instructions,
            plan_agents=agent_dict,
            plan_tools=tool_dict,
            number_of_tasks=n_tasks,
        )
        | (context or {}),
        agents=[agent] if agent else None,
        result_type=list[PlanTask],
        result_validator=validate_plan,
    )

    # create a new flow to avoid polluting the main flow's history
    with Flow():
        plan: list[PlanTask] = plan_task.run()

    task_ids = {}

    for t in plan:
        try:
            task_tools = [tool_dict[i] for i in t.tools]
        except KeyError:
            task_tools = []

        task_ids[t.id] = Task(
            objective=t.objective,
            instructions=t.instructions,
            depends_on=[task_ids[i] for i in t.depends_on],
            parent=task_ids[t.parent] if t.parent else None,
            agents=[agent_dict[a] for a in t.agents] if t.agents else None,
            tools=task_tools,
            context=context or {},
        )

    return list(task_ids.values())
