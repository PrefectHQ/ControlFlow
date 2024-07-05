import logging
import math
from collections import defaultdict
from typing import AsyncGenerator, Generator, Optional, TypeVar, Union

from pydantic import Field, PrivateAttr, field_validator

import controlflow
from controlflow.agents import Agent
from controlflow.events.agent_events import (
    EndTurnEvent,
    SelectAgentEvent,
    SystemMessageEvent,
)
from controlflow.events.events import Event
from controlflow.events.task_events import TaskReadyEvent
from controlflow.flows import Flow
from controlflow.instructions import get_instructions
from controlflow.llm.messages import AIMessage
from controlflow.orchestration.handler import Handler
from controlflow.orchestration.tools import (
    create_end_turn_tool,
    create_task_fail_tool,
    create_task_success_tool,
)
from controlflow.tasks.task import Task
from controlflow.tools import as_tools
from controlflow.tools.tools import Tool
from controlflow.utilities.prefect import prefect_task as prefect_task
from controlflow.utilities.types import ControlFlowModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Controller(ControlFlowModel):
    """
    The controller is responsible for managing the flow of tasks and agents. It
    is given objects that it is responsible for managing. At each iteration, the
    controller will select a task and an agent to complete the task. The
    controller will then create and execute an AgentContext to run the task.
    """

    model_config = dict(arbitrary_types_allowed=True)
    flow: "Flow" = Field(description="The flow that the controller is managing")
    tasks: Optional[list[Task]] = Field(
        None,
        description="Tasks to be completed by the controller. If None, all tasks in the flow will be used.",
    )
    agents: dict[Task, list[Agent]] = Field(
        default_factory=dict,
        description="Optionally assign agents to complete tasks. The provided mapping must be task"
        " -> [agents]. Any tasks that aren't included will use their default agents.",
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
        self, event: Event, tasks: list[Task] = None, agents: list[Agent] = None
    ):
        event.thread_id = self.flow.thread_id
        event.task_ids = [t.id for t in tasks or []]
        event.agent_ids = [a.id for a in agents or []]
        for handler in self.handlers:
            handler.on_event(event)
        if event.persist:
            self.flow.add_events([event])

    def get_agent_context(self, ready_tasks: list[Task]) -> "AgentContext":
        # select an agent
        agent = self.get_agent(ready_tasks=ready_tasks)
        # get ready tasks
        active_tasks = self.get_active_tasks(agent=agent, ready_tasks=ready_tasks)
        # create a context
        context = AgentContext(
            agent=agent,
            tasks=active_tasks,
            flow=self.flow,
            controller=self,
        )
        return context

    def run(self, steps: Optional[int] = None):
        from controlflow.events.controller_events import (
            ControllerEnd,
            ControllerError,
            ControllerStart,
        )

        i = 0
        while any(t.is_incomplete() for t in self.tasks) and i < (steps or math.inf):
            self.handle_event(ControllerStart(controller=self))

            try:
                ready_tasks = self.get_ready_tasks()
                context = self.get_agent_context(ready_tasks=ready_tasks)
                context.run()

            except Exception as exc:
                self.handle_event(ControllerError(controller=self, error=exc))
                raise
            finally:
                self.handle_event(ControllerEnd(controller=self))
            i += 1

    async def run_async(self, steps: Optional[int] = None):
        from controlflow.events.controller_events import (
            ControllerEnd,
            ControllerError,
            ControllerStart,
        )

        i = 0
        while any(t.is_incomplete() for t in self.tasks) and i < (steps or math.inf):
            self.handle_event(ControllerStart(controller=self))

            try:
                ready_tasks = self.get_ready_tasks()
                context = self.get_agent_context(ready_tasks=ready_tasks)
                await context.run_async()

            except Exception as exc:
                self.handle_event(ControllerError(controller=self, error=exc))
                raise
            finally:
                self.handle_event(ControllerEnd(controller=self))
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

    def get_active_tasks(self, agent: Agent, ready_tasks: list[Task]) -> list[Task]:
        """
        Get the subset of ready tasks that the agent is assigned to.
        """
        active_tasks = []
        for task in ready_tasks:
            if agent in self.agents_for_task(task):
                if task._iteration >= (task.max_iterations or math.inf):
                    logger.warning(
                        f'Task "{task.friendly_name()}" has exceeded max iterations and will be marked failed'
                    )
                    task.mark_failed(
                        message="Task was not completed before exceeding its maximum number of iterations."
                    )
                    continue

                task._iteration += 1
                active_tasks.append(task)
                self.handle_event(TaskReadyEvent(task=task), tasks=[task])
                if not task._prefect_task.is_started:
                    task._prefect_task.start(
                        depends_on=[t.result for t in task.depends_on]
                    )
        return active_tasks

    def agents_for_task(self, task: Task) -> list[Agent]:
        return self.agents.get(task, task.get_agents())

    def get_agent(self, ready_tasks: list[Task]) -> tuple[Agent, list[Task]]:
        candidates = [
            agent
            for task in ready_tasks
            # get agents from either controller assignments or the task defaults
            for agent in self.agents_for_task(task)
        ]

        # if there is only one candidate, return it
        if len(candidates) == 1:
            agent = candidates[0]

        # get the last select-agent or end-turn event
        agent_event: list[Union[SelectAgentEvent, EndTurnEvent]] = self.flow.get_events(
            limit=1,
            types=["select-agent", "end-turn"],
            task_ids=[t.id for t in self.flow.graph.upstream_tasks(ready_tasks)],
        )
        if agent_event:
            event = agent_event[0]
            # if an agent was selected and is a candidate, return it
            if event.event == "select-agent":
                agent = next(
                    (a for a in candidates if a.name == event.agent.name), None
                )
                if agent:
                    return agent
            # if an agent was nominated and is a candidate, return it
            elif event.event == "end-turn" and event.next_agent_name is not None:
                agent = next(
                    (a for a in candidates if a.name == event.next_agent_name),
                    None,
                )
                if agent:
                    return agent

        # if there are multiple candiates remaining, use the first task's strategy to select one
        strategy_fn = ready_tasks[0].get_agent_strategy()
        agent = strategy_fn(agents=candidates, task=ready_tasks[0], flow=self.flow)

        self.handle_event(SelectAgentEvent(agent=agent), agents=[agent])
        return agent


class AgentContext(ControlFlowModel):
    agent: Agent = Field(description="The active agent")
    tasks: list[Task] = Field(
        description="The tasks that the agent is assigned to complete that are ready to be completed"
    )
    flow: Flow = Field(description="The flow that the agent is working in")
    controller: Controller = Field(
        description="The controller that is managing the flow"
    )

    def get_events(self) -> list[Event]:
        return self.flow.get_events(
            agent_ids=[self.agent.id],
            task_ids=[t.id for t in self.flow.graph.upstream_tasks(self.tasks)],
        )

    def get_prompt(self, tools: list[Tool]) -> str:
        from controlflow.orchestration import prompts

        # get up to 50 upstream and 50 downstream tasks
        g = self.flow.graph
        upstream_tasks = g.topological_sort([t for t in g.tasks if t.is_complete()])[
            -50:
        ]
        downstream_tasks = g.topological_sort(
            [t for t in g.tasks if t.is_incomplete() and t not in self.tasks]
        )[:50]

        tasks = [t.model_dump() for t in self.tasks]
        upstream_tasks = [t.model_dump() for t in upstream_tasks]
        downstream_tasks = [t.model_dump() for t in downstream_tasks]

        agent_prompt = prompts.AgentTemplate(
            agent=self.agent,
            additional_instructions=get_instructions(),
        )

        workflow_prompt = prompts.WorkflowTemplate(
            flow=self.flow,
            ready_tasks=tasks,
            upstream_tasks=upstream_tasks,
            downstream_tasks=downstream_tasks,
        )

        tool_prompt = prompts.ToolTemplate(
            agent=self.agent,
            has_user_access_tool="talk_to_user" in [t.name for t in tools],
            has_end_turn_tool="end_turn" in [t.name for t in tools],
        )

        communication_prompt = prompts.CommunicationTemplate()

        prompts = [
            p.render()
            for p in [agent_prompt, workflow_prompt, tool_prompt, communication_prompt]
        ]

        return "\n\n".join(prompts)

    def get_tools(self) -> list[Tool]:
        tools = []

        # add flow tools
        tools.extend(self.flow.tools)

        # add end turn tool if there are multiple agents for any task
        if any(len(self.controller.agents_for_task(t)) > 1 for t in self.tasks):
            tools.append(
                create_end_turn_tool(controller=self.controller, agent=self.agent)
            )

        # add tools for working with tasks
        for task in self.tasks:
            tools.extend(task.get_tools())
            tools.append(
                create_task_success_tool(
                    controller=self.controller, task=task, agent=self.agent
                )
            )
            tools.append(
                create_task_fail_tool(
                    controller=self.controller, task=task, agent=self.agent
                )
            )

        return as_tools(tools)

    def get_messages(self, tools: list[Tool] = None) -> list[AIMessage]:
        from controlflow.events.message_compiler import EventContext, MessageCompiler

        events = self.flow.get_events(
            agent_ids=[self.agent.id],
            task_ids=[
                t.id
                for t in self.flow.graph.upstream_tasks(self.tasks)
                if not t.private
            ],
        )

        events.append(
            SystemMessageEvent(content=f"{self.agent.name}, it is your turn.")
        )

        event_context = EventContext(
            llm_rules=self.agent.get_llm_rules(),
            agent=self.agent,
            ready_tasks=self.tasks,
            controller=self.controller,
            flow=self.flow,
        )

        compiler = MessageCompiler(
            events=events,
            context=event_context,
            system_prompt=self.get_prompt(tools=tools),
        )
        messages = compiler.compile_to_messages()
        return messages

    def run(self) -> Generator["Event", None, None]:
        if not self.tasks:
            return
        tools = self.get_tools()
        messages = self.get_messages(tools=tools)
        for event in self.agent._run_model(messages=messages, additional_tools=tools):
            self.controller.handle_event(event, tasks=self.tasks, agents=[self.agent])

    async def run_async(self) -> AsyncGenerator["Event", None]:
        tools = self.get_tools()
        messages = self.get_messages()
        async for event in self.agent._run_model_async(
            messages=messages, additional_tools=tools
        ):
            self.controller.handle_event(event, tasks=self.tasks, agents=[self.agent])
