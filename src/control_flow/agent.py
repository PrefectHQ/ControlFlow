import functools
import inspect
import logging
from typing import Callable, Generic, TypeVar, Union

import marvin
import marvin.utilities.tools
from marvin.beta.assistants.assistants import Assistant
from marvin.beta.assistants.handlers import PrintHandler
from marvin.beta.assistants.runs import Run
from marvin.tools.assistants import AssistantTool, CancelRun
from marvin.types import FunctionTool
from marvin.utilities.asyncio import ExposeSyncMethodsMixin, expose_sync_method
from marvin.utilities.jinja import Environment
from prefect import task as prefect_task
from prefect.artifacts import create_markdown_artifact
from pydantic import BaseModel, Field, field_validator

from control_flow import settings
from control_flow.context import ctx

from .flow import AIFlow
from .task import AITask, TaskStatus

T = TypeVar("T")
logger = logging.getLogger(__name__)


INSTRUCTIONS = """
You are an AI assistant. Your job is to complete the tasks assigned to you.  You
were created by a software application, and any messages you receive are from
that software application, not a user. You may use any tools at your disposal to
complete the task, including talking to a human user.


## Instructions

In addition to completing your tasks, these are your current instructions. You
must follow them at all times, even when using a tool to talk to a user. Note
that instructions can change at any time and the thread history may reflect
different instructions than these:

{% if assistant.instructions -%}
- {{ assistant.instructions }}
{% endif %}
{% if flow.instructions -%}
- {{ flow.instructions }}
{% endif %}
{% if agent.instructions -%}
- {{ agent.instructions }}
{% endif %}
{% for instruction in instructions %}
- {{ instruction }}
{% endfor %}


## Tasks

{% if agent.tasks %}
You have been assigned the following tasks. You will continue to run until all
tasks are finished. It may take multiple attempts, iterations, or tool uses to
complete a task. When a task is finished, mark it as `completed`
(and provide a result, if required) or `failed` (with a brief explanation) by
using the appropriate tool. Do not mark a task as complete if you don't have a
complete result. Do not make up results. If you receive only partial or unclear
information from a user, keep working until you have all the information you
need. Be very sure that a task is truly unsolvable before marking it as failed,
especially when working with a human user.


{% for task in agent.tasks %}
### Task {{ task.id }}
- Status: {{ task.status.value }}
- Objective: {{ task.objective }}
{% if task.status.value == "completed" %}
- Result: {{ task.result }}
{% elif task.status.value == "failed" %}
- Error: {{ task.error }}
{% endif %}
{% if task.context %}
- Context: {{ task.context }}
{% endif %}

{% endfor %}
{% else %}
You have no explicit tasks to complete. Follow your instructions as best as you can.
{% endif %}

## Communication

All messages you receive in the thread are generated by the software that
created you, not a human user. All messages you send are sent only to that
software and are never seen by any human.

{% if agent.system_access -%}
The software that created you is an AI capable of processing natural language,
so you can freely respond by posting messages to the thread.
{% else %}
The software that created you is a Python script that can only process
structured responses produced by your tools. DO NOT POST ANY MESSAGES OR RESPONSES TO THE
THREAD. They will be ignored and only waste time. ONLY USE TOOLS TO RESPOND.
{% endif %}

{% if agent.user_access -%}
There is also a human user who may be involved in the task. You can communicate
with them using the `talk_to_human` tool. The user is a human and unaware of
your tasks or this system. Do not mention your tasks or anything about how the
system works to them. They can only see messages you send them via tool, not the
rest of the thread. When dealing with humans, you may not always get a clear or
correct response. You may need to ask multiple times or rephrase your questions.
{% else %}
You can not communicate with a human user at this time.
{% endif %}


{% if context %}
## Additional context

The following context was provided:
{% for key, value in context.items() -%}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}
"""


RUN_ARTIFACT = """
{% if messages %}
## Messages

{% for message in messages %}
Timestamp: {{ message.created_at }}
Role: {{ message.role }}
Message: {{ message.content[0].text.value }}

{% endfor %}
{% endif %}

## Steps
{% for step in steps %}
```json
{{ step.model_dump_json(indent=2) }}
```

{% endfor %}
"""


def talk_to_human(message: str, get_response: bool = True) -> str:
    """
    Send a message to the human user and optionally wait for a response.
    If `get_response` is True, the function will return the user's response,
    otherwise it will return a simple confirmation.
    """
    print(message)
    if get_response:
        response = input("> ")
        return response
    return "Message sent to user"


def end_run():
    """Use this tool to end the run."""
    raise CancelRun()


class Agent(BaseModel, Generic[T], ExposeSyncMethodsMixin):
    tasks: list[AITask] = []
    flow: AIFlow = Field(None, validate_default=True)
    context: dict = Field(None, validate_default=True)
    user_access: bool = Field(
        None,
        validate_default=True,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    system_access: bool = Field(
        None,
        validate_default=True,
        description="If True, the agent will communicate with the system via messages. "
        "This is usually only used when the agent was spawned by another "
        "agent capable of understanding its responses.",
    )
    assistant: Assistant = None
    tools: list[Union[AssistantTool, Callable]] = []
    instructions: str = None
    model_config: dict = dict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("flow", mode="before")
    def _load_flow_from_ctx(cls, v):
        if v is None:
            v = ctx.get("flow", None)
            if v is None:
                v = AIFlow()
        return v

    @field_validator("context", mode="before")
    def _default_context(cls, v):
        if v is None:
            v = {}
        return v

    @field_validator("user_access", mode="before")
    def _default_user_access(cls, v):
        if v is None:
            v = False
        return v

    @field_validator("system_access", mode="before")
    def _default_system_access(cls, v):
        if v is None:
            v = False
        return v

    def _get_instructions(self, context: dict = None):
        instructions = Environment.render(
            INSTRUCTIONS,
            agent=self,
            flow=self.flow,
            assistant=self.assistant or self.flow.assistant,
            instructions=ctx.get("instructions", []),
            context={**self.context, **(context or {})},
        )

        return instructions

    def _get_tools(self) -> list[AssistantTool]:
        tools = self.flow.tools + self.tools

        if not self.tasks:
            tools.append(end_run)

        for task in self.tasks:
            tools.extend([task._create_complete_tool(), task._create_fail_tool()])

        if self.user_access:
            tools.append(talk_to_human)

        final_tools = []
        for tool in tools:
            if not isinstance(tool, AssistantTool):
                tool = marvin.utilities.tools.tool_from_function(tool)
            if isinstance(tool, FunctionTool):

                async def modified_fn(
                    original_fn=tool.function._python_fn, *args, **kwargs
                ):
                    result = original_fn(*args, **kwargs)
                    await create_markdown_artifact(
                        markdown=f"```json\n{result}\n```", key="result"
                    )
                    return result

                tool.function._python_fn = prefect_task(
                    modified_fn,
                    name=f"Tool call: {tool.function.name}",
                )
            final_tools.append(tool)
        return final_tools

    def _get_openai_run_task(self):
        """
        Helper function for building the task that will execute the OpenAI assistant run.
        This needs to be regenerated each time in case the instructions change.
        """

        @prefect_task(name="Execute OpenAI assistant run")
        async def execute_openai_run(context: dict = None, run_kwargs: dict = None):
            run_kwargs = run_kwargs or {}
            if "model" not in run_kwargs:
                run_kwargs["model"] = settings.assistant_model

            run = Run(
                assistant=self.assistant or self.flow.assistant,
                thread=self.flow.thread,
                instructions=self._get_instructions(context=context),
                additional_tools=self._get_tools(),
                event_handler_class=PrintHandler,
                **run_kwargs,
            )
            await run.run_async()

            await create_markdown_artifact(
                markdown=Environment.render(
                    inspect.cleandoc("""
                        {% for message in run.messages %}
                        ### Message {{ loop.index }}
                        ```json
                        {{message.model_dump_json(indent=2)}}
                        ```
                        
                        {% endfor %}
                        """),
                    run=run,
                ),
                key="messages",
                description="All messages sent and received during the run.",
            )
            await create_markdown_artifact(
                markdown=Environment.render(
                    inspect.cleandoc("""
                        {% for step in run.steps %}
                        ### Step {{ loop.index }}
                        ```json
                        {{step.model_dump_json(indent=2)}}
                        ```
                        
                        {% endfor %}
                        """),
                    run=run,
                ),
                key="steps",
                description="All steps taken during the run.",
            )

        return execute_openai_run

    @expose_sync_method("run")
    async def run_async(self, context: dict = None, **run_kwargs) -> list[AITask]:
        openai_run = self._get_openai_run_task()

        openai_run(context=context, run_kwargs=run_kwargs)

        # if this is not an interactive run, continue to run the AI
        # until all tasks are no longer pending
        if not self.system_access:
            counter = 0
            while (
                any(t.status == TaskStatus.PENDING for t in self.tasks)
                and counter < settings.max_agent_iterations
            ):
                openai_run(context=context, run_kwargs=run_kwargs)
                counter += 1

        result = [t.result for t in self.tasks if t.status == TaskStatus.COMPLETED]

        return result


def ai_task(
    fn=None, *, objective: str = None, user_access: bool = None, **agent_kwargs: dict
):
    """
    Decorator that uses a function to create an AI task. When the function is
    called, an agent is created to complete the task and return the result.
    """

    if fn is None:
        return functools.partial(
            ai_task, objective=objective, user_access=user_access, **agent_kwargs
        )

    sig = inspect.signature(fn)

    if objective is None:
        if fn.__doc__:
            objective = f"{fn.__name__}: {fn.__doc__}"
        else:
            objective = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # first process callargs
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        return run_ai.with_options(name=f"Task: {fn.__name__}")(
            task=objective,
            result_type=fn.__annotations__.get("return"),
            context=bound.arguments,
            user_access=user_access,
            **agent_kwargs,
        )

    return wrapper


def _name_from_objective():
    """Helper function for naming task runs"""
    from prefect.runtime import task_run

    objective = task_run.parameters["task"]
    if len(objective) > 32:
        return f"Task: {objective[:32]}..."
    return f"Task: {objective}"


@prefect_task(task_run_name=_name_from_objective)
def run_ai(
    task: str,
    result_type: T = str,
    context: dict = None,
    user_access: bool = None,
    model: str = None,
    **agent_kwargs: dict,
) -> T:
    """
    Run an agent to complete a task with the given objective and context. The
    response will be of the given result type.
    """

    # load flow
    flow = ctx.get("flow", None)
    if flow is None:
        flow = AIFlow()

    # create task
    ai_task = AITask[result_type](objective=task, context=context)
    flow.add_task(ai_task)

    # run agent
    agent = Agent(tasks=[ai_task], user_access=user_access, flow=flow, **agent_kwargs)
    agent.run(model=model)

    # return
    if ai_task.status == TaskStatus.COMPLETED:
        return ai_task.result
    elif ai_task.status == TaskStatus.FAILED:
        raise ValueError(ai_task.error)
