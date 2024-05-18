from enum import Enum
from typing import Any, Callable, Generic, Literal, Optional, TypeVar, Union

from pydantic import Field

from controlflow.core.agent import Agent
from controlflow.core.task import Task
from controlflow.utilities.types import AssistantTool, ControlFlowModel, FunctionTool

ToolLiteral = TypeVar("ToolLiteral", bound=str)


class SimpleType(Enum):
    NONE = "NONE"
    BOOLEAN = "BOOLEAN"
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    STRING = "STRING"

    def to_type(self):
        return {
            SimpleType.BOOLEAN: bool,
            SimpleType.INTEGER: int,
            SimpleType.FLOAT: float,
            SimpleType.STRING: str,
        }[self]


class ListType(ControlFlowModel):
    list_type: Union[SimpleType, "ListType", "DictType", "UnionType"]

    def to_type(self):
        return list[self.list_type.to_type()]


class DictType(ControlFlowModel):
    key_type: Union[SimpleType, "UnionType"]
    value_type: Union[SimpleType, ListType, "UnionType", "DictType", None]

    def to_type(self):
        return dict[
            self.key_type.to_type(),
            self.value_type.to_type() if self.value_type is not None else None,
        ]


class UnionType(ControlFlowModel):
    union_types: list[Union[SimpleType, ListType, DictType, None]]

    def to_type(self):
        types = [t.to_type() if t is not None else None for t in self.union_types]
        return Union[*types]  # type: ignore


class TaskReference(ControlFlowModel):
    id: int


class AgentReference(ControlFlowModel):
    name: str


class AgentTemplate(ControlFlowModel, Generic[ToolLiteral]):
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
        False, description="Whether the agent can interact with a human user."
    )
    tools: list[ToolLiteral] = Field(
        [], description="The tools that the agent has access to."
    )


class TaskTemplate(ControlFlowModel, Generic[ToolLiteral]):
    id: int
    objective: str = Field(description="The task's objective.")
    instructions: Optional[str] = Field(
        None, description="Instructions for completing the task."
    )
    result_type: Optional[Union[SimpleType, ListType, DictType, UnionType]] = None
    context: dict[str, Union[TaskReference, Any]] = Field(
        default_factory=dict,
        description="The task's context, which can include TaskReferences to create dependencies.",
    )
    depends_on: list[TaskReference] = Field(
        default_factory=list,
        description="Tasks that must be completed before this task can be started.",
    )
    parent: Optional[TaskReference] = Field(
        None, description="The parent task that this task is a subtask of."
    )
    agents: list[AgentReference] = Field(
        default_factory=list,
        description="Any agents assigned to the task. If not specified, the default agent will be used.",
    )
    tools: list[ToolLiteral] = Field(
        [], description="The tools available for this task."
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
        tasks[task_template.id] = Task(
            objective=task_template.objective,
            instructions=task_template.instructions,
            result_type=(
                task_template.result_type.to_type()
                if task_template.result_type
                else None
            ),
            tools=[tools[tool] for tool in task_template.tools],
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


class Templates(ControlFlowModel, Generic[ToolLiteral]):
    task_templates: list[TaskTemplate[ToolLiteral]]
    agent_templates: list[AgentTemplate[ToolLiteral]]


def auto_tasks(
    description: str,
    available_agents: list[Agent] = None,
    available_tools: list[Union[AssistantTool, Callable]] = None,
) -> list[Task]:
    tool_names = []
    for tool in available_tools or []:
        if isinstance(tool, FunctionTool):
            tool_names.append(tool.function.name)
        elif isinstance(tool, AssistantTool):
            tool_names.append(tool.type)
        else:
            tool_names.append(tool.__name__)

    if tool_names:
        literal_tool_names = Literal[*tool_names]  # type: ignore
    else:
        literal_tool_names = None

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
        result_type=Templates[literal_tool_names],
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
