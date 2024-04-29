import json
import logging
from typing import Callable, Self

import prefect
from marvin.beta.assistants import PrintHandler, Run
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from openai.types.beta.threads.runs import ToolCall
from prefect import get_client as get_prefect_client
from prefect import task as prefect_task
from prefect.context import FlowRunContext
from pydantic import BaseModel, Field, field_validator, model_validator

from control_flow.core.agent import Agent
from control_flow.core.flow import Flow
from control_flow.core.task import Task
from control_flow.instructions import get_instructions as get_context_instructions
from control_flow.utilities.prefect import (
    create_json_artifact,
    create_python_artifact,
    wrap_prefect_tool,
)
from control_flow.utilities.types import FunctionTool, Thread

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

    flow: Flow
    agents: list[Agent]
    tasks: list[Task] = Field(
        description="Tasks that the controller will complete.",
        default_factory=list,
    )
    context: dict = {}
    instructions: str = None
    model_config: dict = dict(extra="forbid")

    @field_validator("agents", mode="before")
    def _validate_agents(cls, v):
        if not v:
            raise ValueError("At least one agent is required.")
        return v

    @model_validator(mode="after")
    def _add_tasks_to_flow(self) -> Self:
        for task in self.tasks:
            self.flow.add_task(task)
        return self

    @expose_sync_method("run_agent")
    async def run_agent_async(self, agent: Agent):
        """
        Run the control flow.
        """
        if agent not in self.agents:
            raise ValueError("Agent not found in controller agents.")

        task = await self._get_prefect_run_agent_task(agent)
        await task(agent=agent)

    async def _run_agent(self, agent: Agent, thread: Thread = None) -> Run:
        """
        Run a single agent.
        """
        from control_flow.core.controller.instruction_template import MainTemplate

        instructions_template = MainTemplate(
            agent=agent,
            controller=self,
            context=self.context,
            instructions=get_context_instructions(),
        )

        instructions = instructions_template.render()

        tools = self.flow.tools + agent.get_tools()

        for task in self.tasks:
            task_id = self.flow.get_task_id(task)
            tools = tools + task.get_tools(task_id=task_id)

        # filter tools because duplicate names are not allowed
        final_tools = []
        final_tool_names = set()
        for tool in tools:
            if isinstance(tool, FunctionTool):
                if tool.function.name in final_tool_names:
                    continue
            final_tool_names.add(tool.function.name)
            final_tools.append(wrap_prefect_tool(tool))

        run = Run(
            assistant=agent,
            thread=thread or self.flow.thread,
            instructions=instructions,
            tools=final_tools,
            event_handler_class=AgentHandler,
        )

        await run.run_async()

        return run

    async def _get_prefect_run_agent_task(
        self, agent: Agent, thread: Thread = None
    ) -> Callable:
        @prefect_task(task_run_name=f'Run Agent: "{agent.name}"')
        async def _run_agent(agent: Agent, thread: Thread = None):
            run = await self._run_agent(agent=agent, thread=thread)

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

        return _run_agent

    def task_ids(self) -> dict[Task, int]:
        return {task: self.flow.get_task_id(task) for task in self.tasks}


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
