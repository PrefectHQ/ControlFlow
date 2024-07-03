import logging
from typing import Generator, Optional, TypeVar, Union

from pydantic import Field, field_validator

from controlflow.agents import Agent
from controlflow.events.agent_events import (
    EndTurnEvent,
    SelectAgentEvent,
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

    @field_validator("handlers", mode="before")
    def _handlers(cls, v):
        from controlflow.orchestration.print_handler import PrintHandler

        if v is None:
            v = [PrintHandler()]
        return v

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

    def run_once(self):
        """
        Core pipeline for running the controller.
        """
        from controlflow.events.controller_events import (
            ControllerEnd,
            ControllerError,
            ControllerStart,
        )

        self.handle_event(ControllerStart(controller=self))

        try:
            ready_tasks = self.get_ready_tasks()

            if not ready_tasks:
                return

            # select an agent
            agent = self.get_agent(ready_tasks=ready_tasks)
            active_tasks = self.get_active_tasks(agent=agent, ready_tasks=ready_tasks)

            context = AgentContext(
                agent=agent,
                tasks=active_tasks,
                flow=self.flow,
                controller=self,
            )

            # run
            context.run()

        except Exception as exc:
            self.handle_event(ControllerError(controller=self, error=exc))
            raise
        finally:
            self.handle_event(ControllerEnd(controller=self))

    def run(self):
        while any(t.is_incomplete() for t in self.tasks):
            self.run_once()

    def get_ready_tasks(self) -> list[Task]:
        all_tasks = self.flow.graph.upstream_tasks(self.tasks)
        ready_tasks = [t for t in all_tasks if t.is_ready()]
        return ready_tasks

    def get_active_tasks(self, agent: Agent, ready_tasks: list[Task]) -> list[Task]:
        """
        Get the subset of ready tasks that the agent is assigned to.
        """
        active_tasks = []
        for task in ready_tasks:
            if agent in self.agents.get(task, task.get_agents()):
                active_tasks.append(task)
                self.handle_event(TaskReadyEvent(task=task), tasks=[task])
                if not task._prefect_task.is_started:
                    task._prefect_task.start(
                        depends_on=[t.result for t in task.depends_on]
                    )
        return active_tasks

    def get_agent(self, ready_tasks: list[Task]) -> tuple[Agent, list[Task]]:
        candidates = [
            agent
            for task in ready_tasks
            # get agents from either controller assignments or the task defaults
            for agent in self.agents.get(task, task.get_agents())
        ]

        # if there is only one candidate, return it
        if len(candidates) == 1:
            agent = candidates[0]

        # get the last select-agent event
        select_event: list[Union[SelectAgentEvent, EndTurnEvent]] = (
            self.flow.get_events(limit=1, types=["select-agent", "end-turn"])
        )

        # if an agent was selected and is a candidate, return it
        if select_event and select_event[0].event == "select-agent":
            agent = next(
                (a for a in candidates if a.name == select_event[0].agent.name), None
            )
            if agent:
                return agent
        # if an agent was nominated and is a candidate, return it
        elif select_event and select_event[0].event == "end-turn":
            agent = next(
                (a for a in candidates if a.name == select_event[0].next_agent_name),
                None,
            )
            if agent:
                return agent

        # if there are multiple candiates remaining, use the first task's strategy to select one
        strategy_fn = ready_tasks[0].get_agent_strategy()
        agent = strategy_fn(agents=candidates, task=ready_tasks[0], flow=self.flow)
        ready_tasks[0]._iteration += 1

        self.handle_event(SelectAgentEvent(agent=agent), agents=[agent])
        return agent


class AgentContext(ControlFlowModel):
    agent: Agent
    tasks: list[Task]
    flow: Flow
    controller: Controller

    def get_events(self) -> list[Event]:
        return self.flow.get_events(
            agent_ids=[self.agent.id],
            task_ids=[t.id for t in self.flow.graph.upstream_tasks(self.tasks)],
        )

    def get_prompt(self) -> str:
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

        tool_prompt = prompts.ToolTemplate(agent=self.agent)

        return "\n\n".join(
            [p.render() for p in [agent_prompt, workflow_prompt, tool_prompt]]
        )

    def get_tools(self) -> list[Tool]:
        tools = []

        # add flow tools
        tools.extend(self.flow.tools)

        # add end turn tool
        tools.append(create_end_turn_tool(controller=self.controller, agent=self.agent))

        # add tools for any ready tasks that the agent is assigned to
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

    def get_messages(self) -> list[AIMessage]:
        from controlflow.events.message_compiler import EventContext, MessageCompiler

        events = self.flow.get_events(
            agent_ids=[self.agent.id],
            task_ids=[t.id for t in self.flow.graph.upstream_tasks(self.tasks)],
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
            system_prompt=self.get_prompt(),
        )
        messages = compiler.compile_to_messages()
        return messages

    def run(self) -> Generator["Event", None, None]:
        tools = self.get_tools()
        messages = self.get_messages()
        for event in self.agent._run_model(messages=messages, additional_tools=tools):
            self.controller.handle_event(event, tasks=self.tasks, agents=[self.agent])
