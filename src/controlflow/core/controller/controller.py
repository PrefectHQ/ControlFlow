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
from controlflow.llm.completions import completion, completion_async
from controlflow.llm.handlers import PrintHandler, ResponseHandler, TUIHandler
from controlflow.llm.history import History
from controlflow.llm.messages import AIMessage, MessageType, SystemMessage
from controlflow.tui.app import TUIApp as TUI
from controlflow.utilities.context import ctx
from controlflow.utilities.tasks import all_complete, any_incomplete

logger = logging.getLogger(__name__)


def add_agent_name_to_message(msg: MessageType):
    """
    If the message is from a named assistant, prefix the message with the assistant's name.
    """
    if isinstance(msg, AIMessage) and msg.name:
        msg = msg.model_copy(update={"content": f"{msg.name}: {msg.content}"})
    return msg


class Controller(BaseModel):
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
    _should_stop: bool = False
    _end_turn_counts: dict = PrivateAttr(default_factory=lambda: defaultdict(int))

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

            self._end_turn_counts[key] += 1
            if self._end_turn_counts[key] >= 3:
                self._should_stop = True
                self._end_turn_counts[key] = 0

            return (
                f"Ending turn. {3 - self._end_turn_counts[key]}"
                " more uses will abort the workflow."
            )

        return end_turn

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

    def _run_once_payload(self):
        """
        Generate the payload for a single run of the controller.
        """
        if all(t.is_complete() for t in self.tasks):
            return

        # TODO: show the agent the entire graph, not just immediate upstreams
        # get the tasks to run
        tasks = self.graph.ready_tasks()
        # get the agents
        agent_candidates = [a for t in tasks for a in t.get_agents() if t.is_ready]
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

        from controlflow.core.controller.instruction_template import (
            MainTemplate,
        )

        tools = self.flow.tools + agent.get_tools() + [self._create_end_turn_tool()]

        # add tools for any inactive tasks that the agent is assigned to
        assigned_tools = []
        for task in tasks:
            if agent in task.get_agents():
                assigned_tools.extend(task.get_tools())
        if not assigned_tools:
            raise ValueError(
                f"Agent {agent.name} is not assigned to any of the tasks that are ready to be run."
            )
        tools.extend(assigned_tools)

        # tools = [prefect.task(tool) for tool in tools]

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

        # setup handlers
        handlers = []
        if controlflow.settings.enable_tui:
            handlers.append(TUIHandler())
        if controlflow.settings.enable_print_handler:
            handlers.append(PrintHandler())

        # yield the agent payload
        return dict(
            agent=agent,
            messages=[system_message] + messages,
            tools=tools,
            handlers=handlers,
            # message_preprocessor=add_agent_name_to_message,
        )

    async def run_once_async(self):
        async with self.tui():
            with self.flow:
                payload = self._run_once_payload()
                if payload is not None:
                    agent: Agent = payload.pop("agent")
                    response_handler = ResponseHandler()
                    payload["handlers"].append(response_handler)

                    response_gen = await completion_async(
                        messages=payload["messages"],
                        model=agent.model,
                        tools=payload["tools"],
                        handlers=payload["handlers"],
                        max_iterations=1,
                        # assistant_name=agent.name,
                        # message_preprocessor=payload["message_preprocessor"],
                        stream=True,
                    )
                    async for _ in response_gen:
                        pass

                # save history
                self.history.save_messages(
                    thread_id=self.flow.thread_id,
                    messages=response_handler.response_messages,
                )
        self._iteration += 1

    def run_once(self):
        with self.flow:
            payload = self._run_once_payload()
            if payload is not None:
                agent: Agent = payload.pop("agent")
                response_handler = ResponseHandler()
                payload["handlers"].append(response_handler)

                response_gen = completion(
                    messages=payload["messages"],
                    model=agent.model,
                    tools=payload["tools"],
                    handlers=payload["handlers"],
                    max_iterations=1,
                    # assistant_name=agent.name,
                    # message_preprocessor=payload["message_preprocessor"],
                    stream=True,
                )
                for _ in response_gen:
                    pass

            # save history
            self.history.save_messages(
                thread_id=self.flow.thread_id,
                messages=response_handler.response_messages,
            )
        self._iteration += 1

    async def run_async(self):
        """
        Run the controller until all tasks are complete.
        """
        max_task_iterations = controlflow.settings.max_task_iterations or math.inf
        start_iteration = self._iteration
        if all_complete(self.tasks):
            return
        async with self.tui():
            while any_incomplete(self.tasks) and not self._should_stop:
                await self.run_once_async()
                if self._iteration > start_iteration + max_task_iterations * len(
                    self.tasks
                ):
                    raise ValueError(
                        f"Task iterations exceeded maximum of {max_task_iterations} for each task."
                    )
            self._should_stop = False

    def run(self):
        """
        Run the controller until all tasks are complete.
        """
        max_task_iterations = controlflow.settings.max_task_iterations or math.inf
        start_iteration = self._iteration
        if all_complete(self.tasks):
            return
        while any_incomplete(self.tasks) and not self._should_stop:
            self.run_once()
            if self._iteration > start_iteration + max_task_iterations * len(
                self.tasks
            ):
                raise ValueError(
                    f"Task iterations exceeded maximum of {max_task_iterations} for each task."
                )
        self._should_stop = False
