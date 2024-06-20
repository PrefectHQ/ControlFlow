from enum import Enum
from typing import Any, Callable, Literal, Optional, TypeVar, Union

from pydantic import Field

from controlflow.agents import Agent
from controlflow.tasks.task import Task
from controlflow.utilities.types import ControlFlowModel

ToolLiteral = TypeVar("ToolLiteral", bound=str)


class ResultType(Enum):
    STRING = "STRING"
    NONE = "NONE"


class TaskReference(ControlFlowModel):
    """
    A reference to a task by its ID. Used for indicating task depenencies.
    """

    id: int


class AgentReference(ControlFlowModel):
    """
    A reference to an agent by its name. Used for assigning agents to tasks.
    """

    name: str


class AgentTemplate(ControlFlowModel):
    name: str
    description: Optional[str] = Field(
        None,
        description="A brief description of the agent that will be visible to other agents.",
    )
    instructions: Optional[str] = Field(
        None,
        description="Private instructions for the agent to follow when completing tasks.",
    )
    user_access: bool = Field(
        False, description="If True, the agent can interact with a human user."
    )
    tools: list[str] = Field([], description="The tools that the agent has access to.")


class TaskTemplate(ControlFlowModel):
    id: int
    objective: str = Field(description="The task's objective.")
    instructions: Optional[str] = Field(
        None, description="Instructions for completing the task."
    )
    result_type: Union[ResultType, list[str]] = Field(
        ResultType.STRING,
        description="The type of result expected from the task, defaults to a string output. "
        "Can also be `NONE` if the task does not produce a result (but may have side effects) or "
        "a list of choices if the task has a discrete set of possible outputs.",
    )
    context: dict[str, Union[TaskReference, Any]] = Field(
        default_factory=dict,
        description="The task's context. Values may be constants, TaskReferences, or "
        "collections of either. Any `TaskReferences` will create upstream dependencies, meaning "
        "this task will receive the referenced task's output as input.",
    )
    depends_on: list[TaskReference] = Field(
        default_factory=list,
        description="Tasks that must be completed before this task can be started, "
        "though their outputs are not used.",
    )
    parent: Optional[TaskReference] = Field(
        None,
        description="Indicate that this task is a subtask of a parent. Not required for top-level tasks.",
    )
    agents: list[AgentReference] = Field(
        default_factory=list,
        description="Any agents assigned to the task. If not specified, the default agent will be used.",
    )
    tools: list[str] = Field([], description="The tools available for this task.")
    user_access: bool = Field(
        False, description="If True, the task requires interaction with a human user."
    )


def create_tasks(
    task_templates: list[TaskTemplate],
    agent_templates: list[AgentTemplate] = None,
    agents: list[Agent] = None,
    tools: dict[str, Any] = None,
) -> list[Task]:
    """
    Create tasks from task templates, agent templates, agents, and tools.

    Task templates and agent templates are JSON-serializable objects that define the tasks and agents to be created.

    Agents and tools represent pre-existing agents and tools that can be used to complete the task and agent templates.
    """
    agents: dict[str, Agent] = {a.name: a for a in agents or []}
    tasks: dict[int, Task] = {}
    task_templates: dict[int, TaskTemplate] = {t.id: t for t in task_templates}

    # create agents from templates
    for agent_template in agent_templates:
        agents[agent_template.name] = Agent(
            name=agent_template.name,
            description=agent_template.description,
            instructions=agent_template.instructions,
            user_access=agent_template.user_access,
            tools=[tools[tool] for tool in agent_template.tools],
        )

    # create tasks from templates
    for task_template in task_templates.values():
        if task_template.result_type == ResultType.NONE:
            result_type = None
        elif result_type == ResultType.STRING:
            result_type = str
        else:
            result_type = task_template.result_type

        tasks[task_template.id] = Task(
            objective=task_template.objective,
            instructions=task_template.instructions,
            result_type=result_type,
            tools=[tools[tool] for tool in task_template.tools],
            use_access=task_template.user_access,
        )

    # resolve references
    for template_id, task in tasks.items():
        task_template = task_templates[template_id]
        if task_agents := [
            agents[agent_ref.name] for agent_ref in task_template.agents
        ]:
            task.agents = task_agents
        task.depends_on = [tasks[d.id] for d in task_template.depends_on]
        task.context = {
            key: tasks[value.id] if isinstance(value, TaskReference) else value
            for key, value in task_template.context.items()
        }

        if parent := tasks[task_template.parent.id] if task_template.parent else None:
            parent.add_subtask(task)

    return list(tasks.values())


class Templates(ControlFlowModel):
    task_templates: list[TaskTemplate]
    agent_templates: list[AgentTemplate]


def auto_tasks(
    description: str,
    available_agents: list[Agent] = None,
    available_tools: list[Callable] = None,
) -> list[Task]:
    tool_names = []
    for tool in available_tools or []:
        tool_names.append(tool.__name__)

    if tool_names:
        literal_tool_names = Literal[*tool_names]  # type: ignore
    else:
        literal_tool_names = None

    class TaskTemplate_Tools(TaskTemplate):
        tools: list[literal_tool_names] = Field(
            [], description="The tools available for this task."
        )

    class AgentTemplate_Tools(AgentTemplate):
        tools: list[literal_tool_names] = Field(
            [], description="The tools that the agent has access to."
        )

    class Templates_Tools(Templates):
        task_templates: list[TaskTemplate_Tools]
        agent_templates: list[AgentTemplate_Tools]

    task = Task(
        objective="""
        Generate the minimal set of tasks required to complete the provided
        `description` of an objective. Also reference any tools or agents (or
        create new agents) that your tasks require.
        """,
        instructions="""
        Each task will be executed by an agent like you, working in a workflow
        like this one. Your job is to define the workflow. Choose your tasks to
        be achievable by agents with the tools and skills you deem necessary.
        Create only as many tasks as you need.
        
        Each task should be well-defined, with a single objective and clear
        instructions. The tasks should be independent of each other, but may
        have dependencies on other tasks. If you do not choose
        agents for your tasks, the default agent will be used. Do not post messages, just return your
        result.
        """,
        result_type=Templates,
        context=dict(
            description=description,
            available_agents=available_agents,
            available_tools=available_tools,
        ),
    )

    task.run()

    return create_tasks(
        task_templates=task.result.task_templates,
        agent_templates=task.result.agent_templates,
        agents=available_agents,
        tools=available_tools,
    )
