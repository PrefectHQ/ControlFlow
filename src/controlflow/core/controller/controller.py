import datetime
import json
import logging
import math
from contextlib import asynccontextmanager
from functools import cached_property
from typing import Union

import marvin.utilities
import marvin.utilities.tools
import prefect
from marvin.beta.assistants import EndRun, PrintHandler, Run
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from openai import AsyncAssistantEventHandler
from openai.types.beta.threads import Message, MessageDelta
from openai.types.beta.threads.runs import RunStep, RunStepDelta, ToolCall
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.context import FlowRunContext
from pydantic import BaseModel, Field, computed_field, model_validator

import controlflow
from controlflow.core.agent import Agent
from controlflow.core.controller.moderators import marvin_moderator
from controlflow.core.flow import Flow, get_flow
from controlflow.core.graph import Graph
from controlflow.core.task import Task
from controlflow.instructions import get_instructions
from controlflow.tui.app import TUIApp as TUI
from controlflow.utilities.context import ctx
from controlflow.utilities.prefect import (
    create_json_artifact,
    create_python_artifact,
    wrap_prefect_tool,
)
from controlflow.utilities.tasks import all_complete, any_incomplete
from controlflow.utilities.types import FunctionTool, Thread

logger = logging.getLogger(__name__)


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
    context: dict = {}
    model_config: dict = dict(extra="forbid")
    enable_tui: bool = Field(default_factory=lambda: controlflow.settings.enable_tui)
    _iteration: int = 0
    _should_abort: bool = False
    _endrun_count: int = 0

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

    def _create_help_tool(self) -> FunctionTool:
        @marvin.utilities.tools.tool_from_function
        def help_im_stuck():
            """
            If you are stuck because no tasks are ready to be worked on, you can call this tool to end your turn. A new agent (possibly you) will be selected to go next. If this tool is used 3 times, the workflow will be aborted automatically, so only use it if you are truly stuck.
            """
            self._endrun_count += 1
            if self._endrun_count >= 3:
                self._should_abort = True
                self._endrun_count = 0
            return EndRun()

        return help_im_stuck

    async def _run_agent(
        self, agent: Agent, tasks: list[Task] = None, thread: Thread = None
    ) -> Run:
        """
        Run a single agent.
        """

        @prefect_task(task_run_name=f'Run Agent: "{agent.name}"')
        async def _run_agent(
            controller: Controller,
            agent: Agent,
            tasks: list[Task],
            thread: Thread = None,
        ):
            from controlflow.core.controller.instruction_template import MainTemplate

            tasks = tasks or controller.tasks

            tools = (
                controller.flow.tools
                + agent.get_tools()
                + [controller._create_help_tool()]
            )

            # add tools for any inactive tasks that the agent is assigned to
            for task in tasks:
                if agent in task.get_agents():
                    tools = tools + task.get_tools()

            instructions_template = MainTemplate(
                agent=agent,
                controller=controller,
                tasks=tasks,
                context=controller.context,
                instructions=get_instructions(),
            )
            instructions = instructions_template.render()

            # filter tools because duplicate names are not allowed
            final_tools = []
            final_tool_names = set()
            for tool in tools:
                if isinstance(tool, FunctionTool):
                    if tool.function.name in final_tool_names:
                        continue
                final_tool_names.add(tool.function.name)
                final_tools.append(wrap_prefect_tool(tool))

            handler = TUIHandler if controlflow.settings.enable_tui else AgentHandler

            run = Run(
                assistant=agent,
                thread=thread or controller.flow.thread,
                instructions=instructions,
                tools=final_tools,
                event_handler_class=handler,
            )

            await run.run_async()

            create_json_artifact(
                key="messages",
                data=[m.model_dump() for m in run.messages],
                description="All messages sent and received during the run.",
            )
            create_json_artifact(
                key="actions",
                data=[s.model_dump() for s in run.steps],
                description="All actions taken by the assistant during the run.",
            )
            return run

        return await _run_agent(
            controller=self, agent=agent, tasks=tasks, thread=thread
        )

    def choose_agent(self, agents: list[Agent], tasks: list[Task]) -> Agent:
        return marvin_moderator(
            agents=agents,
            tasks=tasks,
            iteration=self._iteration,
        )

    @asynccontextmanager
    async def tui(self):
        if tui := ctx.get("tui"):
            yield tui
        else:
            tui = TUI(flow=self.flow)
            with ctx(tui=tui):
                async with tui.run_context(run=controlflow.settings.enable_tui):
                    yield tui

    @expose_sync_method("run_once")
    async def run_once_async(self):
        """
        Run the controller for a single iteration of the provided tasks. An agent will be selected to run the tasks.
        """
        async with self.tui():
            with self.flow:
                # get the tasks to run
                ready_tasks = {t for t in self.tasks if t.is_ready()}
                upstreams = {d for t in ready_tasks for d in t.depends_on}
                tasks = list(ready_tasks.union(upstreams))
                # tasks = self.graph.upstream_dependencies(self.tasks, include_tasks=True)

                if all(t.is_complete() for t in tasks):
                    return

                # get the agents
                agent_candidates = {
                    a for t in tasks for a in t.get_agents() if t.is_ready()
                }
                if self.agents:
                    agents = list(agent_candidates.intersection(self.agents))
                else:
                    agents = list(agent_candidates)

                # select the next agent
                if len(agents) == 0:
                    raise ValueError(
                        "No agents were provided that are assigned to tasks that are ready to be run."
                    )
                elif len(agents) == 1:
                    agent = agents[0]
                else:
                    agent = self.choose_agent(agents=agents, tasks=tasks)

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


class TUIHandler(AsyncAssistantEventHandler):
    async def on_message_delta(self, delta: MessageDelta, snapshot: Message) -> None:
        if tui := ctx.get("tui"):
            content = []
            for item in snapshot.content:
                if item.type == "text":
                    content.append(item.text.value)

            tui.update_message(
                m_id=snapshot.id,
                message="\n\n".join(content),
                role=snapshot.role,
                timestamp=datetime.datetime.fromtimestamp(snapshot.created_at),
            )

    async def on_run_step_delta(self, delta: RunStepDelta, snapshot: RunStep) -> None:
        if tui := ctx.get("tui"):
            tui.update_step(snapshot)


class AgentHandler(PrintHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tool_calls = {}

    async def on_tool_call_created(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is created"""

        if tool_call.type == "function":
            task_run_name = "Prepare arguments for tool call"
        else:
            task_run_name = f"Tool call: {tool_call.type}"

        client = get_prefect_client()
        engine_context = FlowRunContext.get()
        if not engine_context:
            return

        task_run = await client.create_task_run(
            task=prefect.Task(fn=lambda: None),
            name=task_run_name,
            extra_tags=["tool-call"],
            flow_run_id=engine_context.flow_run.id,
            dynamic_key=tool_call.id,
            state=prefect.states.Running(),
        )

        self.tool_calls[tool_call.id] = task_run

    async def on_tool_call_done(self, tool_call: ToolCall) -> None:
        """Callback that is fired when a tool call is done"""

        client = get_prefect_client()
        task_run = self.tool_calls.get(tool_call.id)
        if not task_run:
            return
        await client.set_task_run_state(
            task_run_id=task_run.id, state=prefect.states.Completed(), force=True
        )

        # code interpreter is run as a single call, so we can publish a result artifact
        if tool_call.type == "code_interpreter":
            # images = []
            # for output in tool_call.code_interpreter.outputs:
            #     if output.type == "image":
            #         image_path = download_temp_file(output.image.file_id)
            #         images.append(image_path)

            create_python_artifact(
                key="code",
                code=tool_call.code_interpreter.input,
                description="Code executed in the code interpreter",
                task_run_id=task_run.id,
            )
            create_json_artifact(
                key="output",
                data=tool_call.code_interpreter.outputs,
                description="Output from the code interpreter",
                task_run_id=task_run.id,
            )

        elif tool_call.type == "function":
            create_json_artifact(
                key="arguments",
                data=json.dumps(json.loads(tool_call.function.arguments), indent=2),
                description=f"Arguments for the `{tool_call.function.name}` tool",
                task_run_id=task_run.id,
            )
