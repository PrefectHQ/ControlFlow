import logging
import math
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import cached_property
from typing import Callable, Union

from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

import controlflow
from controlflow.core.agent import Agent
from controlflow.core.controller.moderators import classify_moderator
from controlflow.core.flow import Flow, get_flow
from controlflow.core.graph import Graph
from controlflow.core.task import Task
from controlflow.instructions import get_instructions
from controlflow.llm.completions import completion_async
from controlflow.llm.handlers import PrintHandler, TUIHandler
from controlflow.llm.history import History
from controlflow.llm.messages import AssistantMessage, ControlFlowMessage, SystemMessage
from controlflow.tui.app import TUIApp as TUI
from controlflow.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from controlflow.utilities.context import ctx
from controlflow.utilities.tasks import all_complete, any_incomplete

logger = logging.getLogger(__name__)


def add_agent_name_to_message(msg: ControlFlowMessage):
    """
    If the message is from a named assistant, prefix the message with the assistant's name.
    """
    if isinstance(msg, AssistantMessage) and msg.name:
        msg = msg.model_copy(update={"content": f"{msg.name}: {msg.content}"})
    return msg


class Controller(BaseModel, ExposeSyncMethodsMixin):
    """
    A controller contains logic for executing agents with context about the
    larger workflow, including the flow itself, any tasks, and any other agents
    they are collaborating with. The controller is responsible for orchestrating
    agent behavior by generating instructions and tools for each agent. Note
    that while the controller accepts details about (potentially multiple)
    agents and tasks, it's responsiblity is to invoke one agent one time. Other
    mechanisms should be used to orchestrate multiple agents invocations. This
    is done by the controller to avoid tying e.g. agents to tasks or even a
    specific flow.

    """

    # the flow is tracked by the Controller, not the Task, so that tasks can be
    # defined and even instantiated outside a flow. When a Controller is
    # created, we know we're inside a flow context and ready to load defaults
    # and run.
    flow: Flow = Field(
        default_factory=get_flow,
        description="The flow that the controller is a part of.",
        validate_default=True,
    )
    tasks: list[Task] = Field(
        None,
        description="Tasks that the controller will complete.",
    )
    agents: Union[list[Agent], None] = None
    history: History = Field(
        default_factory=controlflow.llm.history.get_default_history
    )
    context: dict = {}
    model_config: dict = dict(extra="forbid")
    enable_tui: bool = Field(default_factory=lambda: controlflow.settings.enable_tui)
    _iteration: int = 0
    _should_abort: bool = False
    _end_run_counts: dict = PrivateAttr(default_factory=lambda: defaultdict(int))

    @computed_field
    @cached_property
    def graph(self) -> Graph:
        return Graph.from_tasks(self.tasks)

    @model_validator(mode="after")
    def _finalize(self):
        if self.tasks is None:
            self.tasks = list(self.flow._tasks.values())
        for task in self.tasks:
            self.flow.add_task(task)
        return self

    def _create_end_turn_tool(self) -> Callable:
        def end_turn():
            """
            Call this tool to skip your turn and let another agent go next. This
            is useful if you are stuck and can not complete any tasks. If this
            tool is used 3 times by any agent the workflow will be aborted
            automatically, so only use it if you are truly stuck and unable to
            proceed.
            """

            # the agent's name is used as the key to track the number of times
            key = getattr(ctx.get("controller_agent", None), "name", None)

            self._end_run_counts[key] += 1
            if self._end_run_counts[key] >= 3:
                self._should_abort = True
                self._end_run_counts[key] = 0

            return (
                f"Ending turn. {3 - self._end_run_counts[key]}"
                " more uses will abort the workflow."
            )

        return end_turn

    async def _run_agent(self, agent: Agent, tasks: list[Task]):
        """
        Run a single agent.
        """

        from controlflow.core.controller.instruction_template import MainTemplate

        tools = self.flow.tools + agent.get_tools() + [self._create_end_turn_tool()]

        # add tools for any inactive tasks that the agent is assigned to
        for task in tasks:
            if agent in task.get_agents():
                tools = tools + task.get_tools()

        instructions_template = MainTemplate(
            agent=agent,
            controller=self,
            tasks=tasks,
            context=self.context,
            instructions=get_instructions(),
        )
        instructions = instructions_template.render()

        # prepare messages
        system_message = SystemMessage(content=instructions)
        messages = self.history.load_messages(thread_id=self.flow.thread_id)

        # setup handler
        if controlflow.settings.enable_tui:
            handlers = [TUIHandler()]
        elif controlflow.settings.enable_print_handler:
            handlers = [PrintHandler()]
        else:
            handlers = []

        # call llm
        response_messages = []
        async for msg in await completion_async(
            messages=[system_message] + messages,
            model=agent.model,
            tools=tools,
            handlers=handlers,
            max_iterations=1,
            assistant_name=agent.name,
            stream=True,
            message_preprocessor=add_agent_name_to_message,
        ):
            response_messages.append(msg)

        # save history
        self.history.save_messages(
            thread_id=self.flow.thread_id, messages=response_messages
        )

        # create_json_artifact(
        #     key="messages",
        #     data=[m.model_dump() for m in run.messages],
        #     description="All messages sent and received during the run.",
        # )
        # create_json_artifact(
        #     key="actions",
        #     data=[s.model_dump() for s in run.steps],
        #     description="All actions taken by the assistant during the run.",
        # )

    def choose_agent(self, agents: list[Agent], tasks: list[Task]) -> Agent:
        return classify_moderator(
            agents=agents,
            tasks=tasks,
            iteration=self._iteration,
        )

    @asynccontextmanager
    async def tui(self):
        if tui := ctx.get("tui"):
            yield tui
        elif controlflow.settings.enable_tui:
            tui = TUI(flow=self.flow)
            with ctx(tui=tui):
                async with tui.run_context():
                    yield tui
        else:
            yield

    @expose_sync_method("run_once")
    async def run_once_async(self):
        """
        Run the controller for a single iteration of the provided tasks. An agent will be selected to run the tasks.
        """
        async with self.tui():
            # put the flow in context
            with self.flow:
                # get the tasks to run
                ready_tasks = {t for t in self.tasks if t.is_ready}
                upstreams = {d for t in ready_tasks for d in t.depends_on}
                tasks = list(ready_tasks.union(upstreams))

                # TODO: show the agent the entire graph, not just immediate upstreams

                if all(t.is_complete() for t in tasks):
                    return

                # get the agents
                agent_candidates = [
                    a for t in tasks for a in t.get_agents() if t.is_ready
                ]
                if len({a.name for a in agent_candidates}) != len(agent_candidates):
                    raise ValueError(
                        "Multiple agents with the same name were found. Agents must have unique names."
                    )
                if self.agents:
                    agents = [a for a in agent_candidates if a in self.agents]
                else:
                    agents = agent_candidates

                # select the next agent
                if len(agents) == 0:
                    raise ValueError(
                        "No agents were provided that are assigned to tasks that are ready to be run."
                    )
                elif len(agents) == 1:
                    agent = agents[0]
                else:
                    agent = self.choose_agent(agents=agents, tasks=tasks)

                with ctx(controller_agent=agent):
                    await self._run_agent(agent, tasks=tasks)

                self._iteration += 1

    @expose_sync_method("run")
    async def run_async(self):
        """
        Run the controller until all tasks are complete.
        """
        max_task_iterations = controlflow.settings.max_task_iterations or math.inf
        start_iteration = self._iteration
        if all_complete(self.tasks):
            return
        async with self.tui():
            while any_incomplete(self.tasks) and not self._should_abort:
                await self.run_once_async()
                if self._iteration > start_iteration + max_task_iterations * len(
                    self.tasks
                ):
                    raise ValueError(
                        f"Task iterations exceeded maximum of {max_task_iterations} for each task."
                    )
            self._should_abort = False
