from typing import Any, Dict, List, Optional, Union

from pydantic import model_validator

from controlflow.agents.agent import Agent
from controlflow.flows import Flow
from controlflow.memory.async_memory import AsyncMemory
from controlflow.memory.memory import Memory
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool
from controlflow.utilities.general import ControlFlowModel
from controlflow.utilities.jinja import prompt_env


class Template(ControlFlowModel):
    model_config = dict(extra="allow")
    template: Optional[str] = None
    template_path: Optional[str] = None

    @model_validator(mode="after")
    def _validate(self):
        if not self.template and not self.template_path:
            raise ValueError("Template or template_path must be provided.")
        return self

    def render(self, **kwargs) -> str:
        if not self.should_render():
            return ""

        render_kwargs = dict(self)
        del render_kwargs["template"]
        del render_kwargs["template_path"]

        if self.template is not None:
            template = prompt_env.from_string(self.template)
        else:
            template = prompt_env.get_template(self.template_path)
        return template.render(**render_kwargs | kwargs)

    def should_render(self) -> bool:
        return True


class AgentTemplate(Template):
    template_path: str = "agent.jinja"
    agent: Agent


class TasksTemplate(Template):
    template_path: str = "tasks.jinja"
    tasks: List[Task]

    def render(self, **kwargs):
        task_hierarchy = build_task_hierarchy(self.tasks)
        return super().render(task_hierarchy=task_hierarchy, **kwargs)

    def should_render(self) -> bool:
        return bool(self.tasks)


class TaskTemplate(Template):
    """
    Template for the active tasks
    """

    template_path: str = "task.jinja"
    task: Task


class FlowTemplate(Template):
    template_path: str = "flow.jinja"
    flow: Flow


class InstructionsTemplate(Template):
    template_path: str = "instructions.jinja"
    instructions: list[str] = []

    def should_render(self) -> bool:
        return bool(self.instructions)


class LLMInstructionsTemplate(Template):
    template_path: str = "llm_instructions.jinja"
    instructions: Optional[list[str]] = None

    def should_render(self) -> bool:
        return bool(self.instructions)


class ToolTemplate(Template):
    template_path: str = "tools.jinja"
    tools: list[Tool]

    def should_render(self) -> bool:
        return any(t.instructions for t in self.tools)


class MemoryTemplate(Template):
    template_path: str = "memories.jinja"
    memories: list[Union[Memory, AsyncMemory]]

    def should_render(self) -> bool:
        return bool(self.memories)


def build_task_hierarchy(provided_tasks: List[Task]) -> List[Dict[str, Any]]:
    """
    Builds a hierarchical structure of tasks, including all descendants of provided tasks
    and their direct ancestors up to the root.

    This function takes a list of tasks and creates a dictionary representation of the
    task hierarchy. It includes all descendants (subtasks) of the provided tasks and
    all direct ancestors up to the root tasks. The resulting structure allows for
    easy traversal of the task hierarchy.

    Args:
        provided_tasks (List[Task]): The initial list of tasks to build the hierarchy from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the root tasks.
        Each dictionary contains the following keys:
            - 'task': The Task object
            - 'children': A list of child task dictionaries (recursively structured)
            - 'is_active': Boolean indicating if the task was in the original provided_tasks list

    Example:
        If task A has subtasks B and C, and B has a subtask D, calling
        build_task_hierarchy([B]) would return a structure representing:
        A (is_active: False)
        └── B (is_active: True)
            ├── C (is_active: False)
            └── D (is_active: False)
    """
    task_dict = {}
    active_tasks = set(provided_tasks)

    def collect_descendants(task: Task):
        """Recursively collects all descendants of a task."""
        if task not in task_dict:
            task_dict[task] = {
                "task": task,
                "children": [],
                "is_active": task in active_tasks,
            }
        for subtask in task.subtasks:
            if subtask not in task_dict:
                collect_descendants(subtask)
            task_dict[task]["children"].append(task_dict[subtask])

    def collect_ancestors(task: Task):
        """Recursively collects all direct ancestors of a task."""
        if task.parent and task.parent not in task_dict:
            task_dict[task.parent] = {
                "task": task.parent,
                "children": [],
                "is_active": task.parent in active_tasks,
            }
            collect_ancestors(task.parent)
            task_dict[task.parent]["children"].append(task_dict[task])

    # First pass: Collect all descendants of the provided tasks
    for task in provided_tasks:
        collect_descendants(task)

    # Second pass: Collect all direct ancestors of the provided tasks
    for task in provided_tasks:
        collect_ancestors(task)

    # Get root tasks (those without parents or whose parents are not in the task list)
    roots = [
        v for k, v in task_dict.items() if not k.parent or k.parent not in task_dict
    ]

    # Sort children by creation time
    def sort_children(task_info: dict[str, Any]):
        task_info["children"].sort(key=lambda x: x["task"].created_at)
        for child in task_info["children"]:
            sort_children(child)

    for root in roots:
        sort_children(root)

    return roots
