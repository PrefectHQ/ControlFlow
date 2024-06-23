import inspect
import logging
import math
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Callable, Union

from pydantic import Field, PrivateAttr, model_validator

import controlflow
from controlflow.agents import Agent
from controlflow.controllers.graph import Graph
from controlflow.controllers.messages import prepare_messages
from controlflow.flows import Flow, get_flow
from controlflow.instructions import get_instructions
from controlflow.llm.completions import completion, completion_async
from controlflow.llm.handlers import PrintHandler, ResponseHandler, TUIHandler
from controlflow.llm.messages import MessageType, SystemMessage
from controlflow.llm.tools import as_tools
from controlflow.tasks.task import Task
from controlflow.tui.app import TUIApp as TUI
from controlflow.utilities.context import ctx
from controlflow.utilities.prefect import create_markdown_artifact
from controlflow.utilities.prefect import task as prefect_task
from controlflow.utilities.tasks import all_complete, any_incomplete
from controlflow.utilities.types import ControlFlowModel

logger = logging.getLogger(__name__)


def create_messages_markdown_artifact(messages, thread_id):
    markdown_messages = "\n\n".join([f"{msg.role}: {msg.content}" for msg in messages])
    create_markdown_artifact(
        key="messages",
        markdown=inspect.cleandoc(
            """
            # Messages
            
            *Thread ID: {thread_id}*
            
            {markdown_messages}
            """.format(
                thread_id=thread_id,
                markdown_messages=markdown_messages,
            )
        ),
    )


class Controller(ControlFlowModel):
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
    context: dict = {}
    model_config: dict = dict(extra="forbid")
    enable_tui: bool = Field(default_factory=lambda: controlflow.settings.enable_tui)
    _iteration: int = 0
    _should_stop: bool = False
    _end_turn_counts: dict = PrivateAttr(default_factory=lambda: defaultdict(int))

    @property
    def graph(self) -> Graph:
        return Graph.from_tasks(self.tasks)

    @model_validator(mode="after")
    def _finalize(self):
        if self.tasks is None:
            self.tasks = list(self.flow.tasks.values())
        for task in self.tasks:
            self.flow.add_task(task)
        return self

    def _create_end_turn_tool(self) -> Callable:
        def emergency_end_turn():
            """
            This tool is for emergencies only; you should not use it normally.
            If you find yourself in a situation where you are repeatedly invoked
            and your normal tools do not work, or you can not escape the loop,
            use this tool to signal to the controller that you are stuck. A new
            agent will be selected to go next. If this tool is used 3 times by
            an agent the workflow will be aborted automatically.

            """

            # the agent's name is used as the key to track the number of times
            key = getattr(ctx.get("agent", None), "name", None)

            self._end_turn_counts[key] += 1
            if self._end_turn_counts[key] >= 3:
                self._should_stop = True
                self._end_turn_counts[key] = 0

            return (
                f"Ending turn. {3 - self._end_turn_counts[key]}"
                " more uses will abort the workflow."
            )

        return emergency_end_turn

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

    def _setup_run(self):
        """
        Generate the payload for a single run of the controller.
        """
        # TODO: show the agent the entire graph, not just immediate upstreams
        tasks = self.graph.topological_sort()
        ready_tasks = [t for t in tasks if t.is_ready]

        # start tracking tasks
        for task in ready_tasks:
            if not task._prefect_task.is_started:
                task._prefect_task.start(
                    depends_on=[
                        t.result for t in task.depends_on if t.result is not None
                    ]
                )

        # if there are no ready tasks, return. This will usually happen because
        # all the tasks are complete.
        if not ready_tasks:
            return

        # get an agent from the next ready task
        agents = ready_tasks[0].get_agents()
        if len(agents) != 1:
            strategy_fn = ready_tasks[0].get_agent_strategy()
            agent = strategy_fn(agents=agents, task=ready_tasks[0], flow=self.flow)
            ready_tasks[0]._iteration += 1
        else:
            agent = agents[0]

        from controlflow.controllers.instruction_template import MainTemplate

        tools = self.flow.tools + agent.get_tools() + [self._create_end_turn_tool()]

        # add tools for any ready tasks that the agent is assigned to
        for task in ready_tasks:
            if agent in task.get_agents():
                tools.extend(task.get_tools())

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
        messages = self.flow.get_messages()

        rules = agent.get_llm_rules()
        messages = prepare_messages(
            agent=agent,
            system_message=system_message,
            messages=messages,
            rules=rules,
            tools=tools,
        )

        # setup handlers
        handlers = []
        if controlflow.settings.enable_tui:
            handlers.append(TUIHandler())
        if controlflow.settings.enable_print_handler:
            handlers.append(PrintHandler())
        # yield the agent payload
        return dict(
            agent=agent,
            messages=messages,
            tools=as_tools(tools),
            handlers=handlers,
        )

    @prefect_task(task_run_name="Run LLM")
    async def run_once_async(self) -> list[MessageType]:
        async with self.tui():
            payload = self._setup_run()
            if payload is None:
                return
            agent: Agent = payload.pop("agent")
            response_handler = ResponseHandler()
            payload["handlers"].append(response_handler)

            with ctx(agent=agent, flow=self.flow, controller=self):
                response_gen = await completion_async(
                    messages=payload["messages"],
                    model=agent.get_model(),
                    tools=payload["tools"],
                    handlers=payload["handlers"],
                    max_iterations=1,
                    stream=True,
                    agent=agent,
                )
                async for _ in response_gen:
                    pass

            # save history
            self.flow.add_messages(
                messages=response_handler.response_messages,
            )
            self._iteration += 1

            create_messages_markdown_artifact(
                messages=response_handler.response_messages,
                thread_id=self.flow.thread_id,
            )

        return response_handler.response_messages

    @prefect_task(task_run_name="Run LLM")
    def run_once(self) -> list[MessageType]:
        payload = self._setup_run()
        if payload is None:
            return
        agent: Agent = payload.pop("agent")
        response_handler = ResponseHandler()
        payload["handlers"].append(response_handler)

        with ctx(
            agent=agent,
            flow=self.flow,
            controller=self,
        ):
            response_gen = completion(
                messages=payload["messages"],
                model=agent.get_model(),
                tools=payload["tools"],
                handlers=payload["handlers"],
                max_iterations=1,
                stream=True,
                agent=agent,
            )
            for _ in response_gen:
                pass

        # save history
        self.flow.add_messages(
            messages=response_handler.response_messages,
        )
        self._iteration += 1

        create_messages_markdown_artifact(
            messages=response_handler.response_messages,
            thread_id=self.flow.thread_id,
        )

        return response_handler.response_messages

    @prefect_task(task_run_name="Run LLM Controller")
    async def run_async(self) -> list[MessageType]:
        """
        Run the controller until all tasks are complete.
        """
        max_task_iterations = controlflow.settings.max_task_iterations or math.inf
        start_iteration = self._iteration
        if all_complete(self.tasks):
            return

        messages = []
        async with self.tui():
            while any_incomplete(self.tasks) and not self._should_stop:
                new_messages = await self.run_once_async()
                messages.extend(new_messages)
                if self._iteration > start_iteration + max_task_iterations * len(
                    self.tasks
                ):
                    raise ValueError(
                        f"Task iterations exceeded maximum of {max_task_iterations} for each task."
                    )
            self._should_stop = False
            return messages

    @prefect_task(task_run_name="Run LLM Controller")
    def run(self) -> list[MessageType]:
        """
        Run the controller until all tasks are complete.
        """
        max_task_iterations = controlflow.settings.max_task_iterations or math.inf
        start_iteration = self._iteration
        if all_complete(self.tasks):
            return

        messages = []
        while any_incomplete(self.tasks) and not self._should_stop:
            new_messages = self.run_once()
            messages.extend(new_messages)
            if self._iteration > start_iteration + max_task_iterations * len(
                self.tasks
            ):
                raise ValueError(
                    f"Task iterations exceeded maximum of {max_task_iterations} for each task."
                )
        self._should_stop = False
        return messages
