import logging
import math
from typing import Optional, TypeVar

from pydantic import Field, field_validator

import controlflow
from controlflow.agents.agent import Agent
from controlflow.events.base import Event
from controlflow.events.message_compiler import MessageCompiler
from controlflow.flows import Flow
from controlflow.instructions import get_instructions
from controlflow.llm.messages import BaseMessage
from controlflow.orchestration.handler import Handler
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, tool
from controlflow.utilities.general import ControlFlowModel
from controlflow.utilities.prefect import prefect_task as prefect_task

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["Orchestrator"]


class Orchestrator(ControlFlowModel):
    """
    The orchestrator is responsible for managing the flow of tasks and agents.
    It is given tasks to execute in a flow context, and an agent to execute the
    tasks. If other agents are assigned to any of the tasks, the orchestrator
    will provide tools for delegating.
    """

    model_config = dict(arbitrary_types_allowed=True)
    flow: "Flow" = Field(description="The flow that the orchestrator is managing")
    agent: Agent = Field(description="The currently active agent")
    tasks: list[Task] = Field(description="Tasks to be executed by the agent.")
    handlers: list[Handler] = Field(None, validate_default=True)

    @field_validator("handlers", mode="before")
    def _handlers(cls, v):
        from controlflow.orchestration.print_handler import PrintHandler

        if v is None and controlflow.settings.enable_print_handler:
            v = [PrintHandler()]
        return v or []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for task in self.tasks:
            self.flow.add_task(task)

    def handle_event(self, event: Event):
        for handler in self.handlers:
            handler.handle(event)
        if event.persist:
            self.flow.add_events([event])

    def set_agent(self, agent: Agent):
        """Set the current active agent."""
        self.agent = agent

    def get_tools(self) -> list[Tool]:
        tools = []
        tools.extend(self.flow.tools)
        for task in self.get_tasks("assigned"):
            tools.extend(task.get_tools())
        if len(self.get_available_agents() - {self.agent}) > 0:
            tools.append(self.create_delegation_tool())
        return tools

    def get_available_agents(self) -> set[Agent]:
        active_tasks = self.get_tasks("active")
        return set(a for t in active_tasks for a in t.get_agents())

    def create_delegation_tool(self) -> Tool:
        @tool
        def delegate_to_agent(agent_id: str) -> str:
            """
            Select another agent to take the next turn.
            """
            available_agents = {a.id: a for a in self.get_available_agents()}
            if agent_id not in available_agents:
                raise ValueError(
                    f"Agent with ID {agent_id} not found or not available."
                )
            self.set_agent(available_agents[agent_id])
            return f"Delegated to agent {agent_id}"

        return delegate_to_agent

    def run(self, steps: Optional[int] = None):
        from controlflow.events.orchestrator_events import (
            OrchestratorEnd,
            OrchestratorError,
            OrchestratorStart,
        )

        i = 0
        while self.get_tasks("active") and i < (steps or math.inf):
            breakpoint()
            self.handle_event(OrchestratorStart(orchestrator=self))

            try:
                messages = self.compile_messages()
                for event in self.agent._run_model(
                    messages=messages, tools=self.get_tools()
                ):
                    self.handle_event(event)

            except Exception as exc:
                self.handle_event(OrchestratorError(orchestrator=self, error=exc))
                raise
            finally:
                self.handle_event(OrchestratorEnd(orchestrator=self))
                i += 1

    def compile_prompt(self) -> str:
        from controlflow.orchestration.prompt_templates import (
            InstructionsTemplate,
            TasksTemplate,
            ToolTemplate,
        )

        tools = self.get_tools()

        prompts = [
            self.agent.get_prompt(),
            self.flow.get_prompt(),
            TasksTemplate(tasks=self.get_tasks("active")).render(),
            ToolTemplate(tools=tools).render(),
            InstructionsTemplate(instructions=get_instructions()).render(),
        ]
        prompt = "\n\n".join([p for p in prompts if p])
        return prompt

    def compile_messages(self) -> list[BaseMessage]:
        events = self.flow.get_events(limit=100)

        compiler = MessageCompiler(
            events=events,
            llm_rules=self.agent.get_llm_rules(),
            system_prompt=self.compile_prompt(),
        )
        messages = compiler.compile_to_messages(agent=self.agent)
        return messages

    def get_tasks(self, filter: str = "assigned") -> list[Task]:
        """
        Collect tasks based on the specified filter.

        Args:
            filter (str): Determines which tasks to return.
                - "active": Tasks ready to execute (no unmet dependencies).
                - "assigned": Active tasks assigned to the current agent.
                - "all": All tasks including subtasks and ancestors.

        Returns:
            list[Task]: List of tasks based on the specified filter.
        """
        if filter not in ["active", "assigned", "all"]:
            raise ValueError(f"Invalid filter: {filter}")

        all_tasks: list[Task] = []
        active_tasks: list[Task] = []

        def collect_tasks(task: Task, is_root: bool = False):
            if task not in all_tasks:
                all_tasks.append(task)
                if is_root and task.is_ready():
                    active_tasks.append(task)
                for subtask in task.subtasks:
                    collect_tasks(subtask, is_root=is_root)

        # Collect tasks from self.tasks (root tasks)
        for task in self.tasks:
            collect_tasks(task, is_root=True)

        if filter == "active":
            return active_tasks

        if filter == "assigned":
            return [task for task in active_tasks if self.agent in task.get_agents()]

        # Collect ancestor tasks for "all" filter
        for task in self.tasks:
            current = task.parent
            while current:
                if current not in all_tasks:
                    all_tasks.append(current)
                current = current.parent

        return all_tasks

    def get_task_hierarchy(self) -> dict:
        """
        Build a hierarchical structure of all tasks.

        Returns:
            dict: A nested dictionary representing the task hierarchy,
            where each task has 'task' and 'children' keys.
        """
        all_tasks = self.get_tasks("all")

        hierarchy = {}
        task_dict_map = {task.id: {"task": task, "children": []} for task in all_tasks}

        for task in all_tasks:
            if task.parent:
                parent_dict = task_dict_map[task.parent.id]
                parent_dict["children"].append(task_dict_map[task.id])
            else:
                hierarchy[task.id] = task_dict_map[task.id]

        return hierarchy
