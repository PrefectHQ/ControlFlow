import logging
import math
from collections import defaultdict
from typing import Optional, TypeVar

from pydantic import Field, PrivateAttr, field_validator

import controlflow
from controlflow.agents.agent import Agent, BaseAgent
from controlflow.events.base import Event
from controlflow.flows import Flow
from controlflow.orchestration.agent_context import AgentContext
from controlflow.orchestration.handler import Handler
from controlflow.tasks.task import Task
from controlflow.tools.orchestration import (
    create_task_fail_tool,
    create_task_success_tool,
)
from controlflow.tools.tools import Tool
from controlflow.utilities.prefect import prefect_task as prefect_task
from controlflow.utilities.types import ControlFlowModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["Orchestrator"]


class Orchestrator(ControlFlowModel):
    """
    The orchestrator is responsible for managing the flow of tasks and agents. It
    is given objects that it is responsible for managing. At each iteration, the
    orchestrator will select a task and an agent to complete the task.
    """

    model_config = dict(arbitrary_types_allowed=True)
    flow: "Flow" = Field(description="The flow that the orchestrator is managing")
    tasks: Optional[list[Task]] = Field(
        None,
        description="Target tasks to be completed by the orchestrator. "
        "Note that any upstream dependencies will be completed as well. "
        "If None, all tasks in the flow will be used.",
    )
    agents: dict[Task, BaseAgent] = Field(
        default_factory=dict,
        description="Optionally assign an agent to a task; this overrides the task's own configuration.",
    )
    handlers: list[Handler] = Field(None, validate_default=True)

    _task_iterations: dict[Task, int] = PrivateAttr(
        default_factory=lambda: defaultdict(int)
    )
    _ready_task_counter: int = 0

    @field_validator("handlers", mode="before")
    def _handlers(cls, v):
        from controlflow.orchestration.print_handler import PrintHandler

        if v is None and controlflow.settings.enable_print_handler:
            v = [PrintHandler()]
        return v or []

    @field_validator("agents", mode="before")
    def _agents(cls, v):
        if v is None:
            v = {}
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = self.tasks or self.flow.tasks
        for task in self.tasks:
            self.flow.add_task(task)

    def handle_event(
        self, event: Event, tasks: list[Task] = None, agent: BaseAgent = None
    ):
        event.thread_id = self.flow.thread_id
        event.add_tasks(tasks or [])
        event.add_agents([agent] if agent else [])
        for handler in self.handlers:
            handler.handle(event)
        if event.persist:
            self.flow.add_events([event])

    def run(self, steps: Optional[int] = None):
        from controlflow.events.orchestrator_events import (
            OrchestratorEnd,
            OrchestratorError,
            OrchestratorStart,
        )

        i = 0
        while any(t.is_incomplete() for t in self.tasks) and i < (steps or math.inf):
            self.handle_event(OrchestratorStart(orchestrator=self))

            try:
                ready_tasks = self.get_ready_tasks()
                if not ready_tasks:
                    return
                agent = self.get_agent(task=ready_tasks[0])
                tasks = self.get_agent_tasks(agent=agent, ready_tasks=ready_tasks)
                tools = self.get_tools(tasks=tasks)

                context = AgentContext(
                    agent=agent,
                    flow=self.flow,
                    tasks=tasks,
                    tools=tools,
                    handlers=self.handlers,
                )
                with context:
                    agent._run(context=context)

            except Exception as exc:
                self.handle_event(OrchestratorError(orchestrator=self, error=exc))
                raise
            finally:
                self.handle_event(OrchestratorEnd(orchestrator=self))
                i += 1

    def get_ready_tasks(self) -> list[Task]:
        all_tasks = self.flow.graph.upstream_tasks(self.tasks)
        ready_tasks = [t for t in all_tasks if t.is_ready()]
        if not ready_tasks:
            self._ready_task_counter += 1
            if self._ready_task_counter >= 3:
                raise ValueError("No tasks are ready to run. This is unexpected.")
        else:
            self._ready_task_counter = 0
        return ready_tasks

    def get_agent_tasks(self, agent: BaseAgent, ready_tasks: list[Task]) -> list[Task]:
        """
        Get the subset of ready tasks that the agent is assigned to.
        """
        agent_tasks = []
        for task in ready_tasks:
            if agent is self.get_agent(task):
                if task._iteration >= (task.max_iterations or math.inf):
                    logger.warning(
                        f'Task "{task.friendly_name()}" has exceeded max iterations and will be marked failed'
                    )
                    task.mark_failed(
                        message="Task was not completed before exceeding its maximum number of iterations."
                    )
                    continue

                # if the task is pending, start it
                if task.is_pending():
                    task.mark_running()
                    # self.handle_event(
                    #     ActivateAgent(
                    #         agent=agent, content=agent.get_activation_prompt()
                    #     ),
                    #     agent=agent,
                    #     tasks=[task],
                    # )

                task._iteration += 1
                agent_tasks.append(task)
        return agent_tasks

    def get_agent(self, task: Task) -> Agent:
        if task in self.agents:
            return self.agents[task]
        else:
            return task.get_agent()

    def get_tools(self, tasks: list[Task]) -> list[Tool]:
        tools = []
        tools.extend(self.flow.tools)
        for task in tasks:
            tools.extend(task.get_tools())
            tools.append(create_task_success_tool(task=task))
            tools.append(create_task_fail_tool(task=task))
        return tools
