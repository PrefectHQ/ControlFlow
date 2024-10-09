import abc
import logging
import random
import warnings
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Optional,
    Union,
)

from langchain_core.language_models import BaseChatModel
from pydantic import Field, field_serializer, field_validator
from typing_extensions import Self

import controlflow
from controlflow.agents.names import AGENT_NAMES
from controlflow.events.base import Event
from controlflow.instructions import get_instructions
from controlflow.llm.messages import AIMessage, BaseMessage
from controlflow.llm.models import get_model as get_model_from_string
from controlflow.llm.rules import LLMRules
from controlflow.memory import Memory
from controlflow.tools.tools import (
    Tool,
    as_lc_tools,
    as_tools,
    handle_tool_call,
    handle_tool_call_async,
)
from controlflow.utilities.context import ctx
from controlflow.utilities.general import ControlFlowModel, hash_objects, unwrap
from controlflow.utilities.prefect import create_markdown_artifact, prefect_task

if TYPE_CHECKING:
    from controlflow.orchestration.handler import Handler
    from controlflow.orchestration.turn_strategies import TurnStrategy
    from controlflow.tasks import Task
    from controlflow.tools.tools import Tool
logger = logging.getLogger(__name__)


class Agent(ControlFlowModel, abc.ABC):
    """
    Class for objects that can be used as agents in a flow
    """

    model_config = dict(arbitrary_types_allowed=True)

    id: str = Field(None)
    name: str = Field(
        default_factory=lambda: random.choice(AGENT_NAMES),
        description="The name of the agent.",
    )
    description: Optional[str] = Field(
        None, description="A description of the agent, visible to other agents."
    )
    instructions: Optional[str] = Field(
        "You are a diligent AI assistant. You complete your tasks efficiently and without error.",
        description="Instructions for the agent, private to this agent.",
    )
    prompt: Optional[str] = Field(
        None,
        description="A system template for the agent. The template should be formatted as a jinja2 template.",
    )
    tools: list[Tool] = Field([], description="List of tools available to the agent.")
    interactive: bool = Field(
        False,
        description="If True, the agent is given tools for interacting with a human user.",
    )
    memories: list[Memory] = Field(
        default=[],
        description="A list of memory modules for the agent to use.",
    )

    model: Optional[Union[str, BaseChatModel]] = Field(
        None,
        description="The LangChain BaseChatModel used by the agent. If not provided, the default model will be used. A compatible string can be passed to automatically retrieve the model.",
        exclude=True,
    )
    llm_rules: Optional[LLMRules] = Field(
        None,
        description="The LLM rules for the agent. If not provided, the rules will be inferred from the model (if possible).",
    )

    _cm_stack: list[contextmanager] = []

    def __init__(self, instructions: str = None, **kwargs):
        if instructions is not None:
            kwargs["instructions"] = instructions

        # deprecated in 0.9
        if "user_access" in kwargs:
            warnings.warn(
                "The `user_access` argument is deprecated. Use `interactive=True` instead.",
                DeprecationWarning,
            )
            kwargs["interactive"] = kwargs.pop("user_access")

        if additional_instructions := get_instructions():
            kwargs["instructions"] = (
                kwargs.get("instructions")
                or "" + "\n" + "\n".join(additional_instructions)
            ).strip()

        super().__init__(**kwargs)

        if not self.id:
            self.id = self._generate_id()

    def __hash__(self) -> int:
        return id(self)

    def _generate_id(self):
        """
        Helper function to generate a stable, short, semi-unique ID for the agent.
        """
        return hash_objects(
            (
                type(self).__name__,
                self.name,
                self.description,
                self.prompt,
                self.instructions,
            )
        )

    @field_validator("instructions")
    def _validate_instructions(cls, v):
        if v:
            v = unwrap(v)
        return v

    @field_validator("tools", mode="before")
    def _validate_tools(cls, tools: list[Tool]):
        return as_tools(tools or [])

    @field_validator("model", mode="before")
    def _validate_model(cls, model: Optional[Union[str, BaseChatModel]]):
        if isinstance(model, str):
            return get_model_from_string(model)
        return model

    @field_serializer("tools")
    def _serialize_tools(self, tools: list[Tool]):
        tools = controlflow.tools.as_tools(tools)
        return [t.model_dump(include={"name", "description"}) for t in tools]

    def serialize_for_prompt(self) -> dict:
        dct = self.model_dump(
            include={"name", "id", "description", "tools", "interactive"}
        )
        if not dct["interactive"]:
            dct.pop("interactive")
        return dct

    def get_model(self, tools: list["Tool"] = None) -> BaseChatModel:
        """
        Retrieve the LLM model for this agent
        """
        model = self.model or controlflow.defaults.model
        if model is None:
            raise ValueError(
                f"Agent {self.name}: No model provided and no default model could be loaded."
            )
        if tools:
            model = model.bind_tools(as_lc_tools(tools))
        return model

    def get_llm_rules(self) -> LLMRules:
        """
        Retrieve the LLM rules for this agent's model
        """
        if self.llm_rules is None:
            return controlflow.llm.rules.rules_for_model(self.get_model())
        else:
            return self.llm_rules

    def get_tools(self) -> list["Tool"]:
        from controlflow.tools.input import cli_input

        tools = self.tools.copy()
        if self.interactive:
            tools.append(cli_input)
        for memory in self.memories:
            tools.extend(memory.get_tools())

        return as_tools(tools)

    def get_prompt(self) -> str:
        from controlflow.orchestration import prompt_templates

        template = prompt_templates.AgentTemplate(template=self.prompt, agent=self)
        return template.render()

    @contextmanager
    def create_context(self) -> Generator[Self, None, None]:
        with ctx(agent=self):
            yield self

    def __enter__(self) -> Self:
        self._cm_stack.append(self.create_context())
        return self._cm_stack[-1].__enter__()

    def __exit__(self, *exc_info):
        return self._cm_stack.pop().__exit__(*exc_info)

    def run(
        self,
        objective: str,
        *,
        turn_strategy: "TurnStrategy" = None,
        handlers: list["Handler"] = None,
        **task_kwargs,
    ):
        return controlflow.run(
            objective=objective,
            agents=[self],
            turn_strategy=turn_strategy,
            handlers=handlers,
            **task_kwargs,
        )

    async def run_async(
        self,
        objective: str,
        *,
        turn_strategy: "TurnStrategy" = None,
        handlers: list["Handler"] = None,
        **task_kwargs,
    ):
        return await controlflow.run_async(
            objective=objective,
            agents=[self],
            turn_strategy=turn_strategy,
            handlers=handlers,
            **task_kwargs,
        )

    def plan(
        self,
        objective: str,
        instructions: Optional[str] = None,
        agents: Optional[list["Agent"]] = None,
        tools: Optional[list["Tool"]] = None,
        context: Optional[dict] = None,
    ) -> list["Task"]:
        """
        Generate a list of tasks that represent a structured plan for achieving
        the objective.

        Args:
            objective (str): The objective to plan for.
            instructions (Optional[str]): Optional instructions for the planner.
            agents (Optional[list[Agent]]): Optional list of agents to include in the plan. If None, this agent is used.
            tools (Optional[list[Tool]]): Optional list of tools to include in the plan. If None, this agent's tools are used.
            context (Optional[dict]): Optional context to include in the plan.

        Returns:
            list[Task]: A list of tasks that represent a structured plan for achieving the objective.
        """
        return controlflow.tasks.plan(
            objective=objective,
            instructions=instructions,
            agent=self,
            agents=agents or [self],
            tools=tools or [self.tools],
            context=context,
        )

    @prefect_task(task_run_name="Call LLM")
    def _run_model(
        self,
        messages: list[BaseMessage],
        tools: list["Tool"],
        stream: bool = True,
        model_kwargs: Optional[dict] = None,
    ) -> Generator[Event, None, None]:
        from controlflow.events.events import (
            AgentMessage,
            AgentMessageDelta,
            ToolCallEvent,
            ToolResultEvent,
        )

        tools = as_tools(self.get_tools() + tools)
        model = self.get_model(tools=tools)

        logger.debug(f"Running model {model} for agent {self.name} with tools {tools}")
        if controlflow.settings.log_all_messages:
            logger.debug(f"Input messages: {messages}")

        if stream:
            response = None
            for delta in model.stream(messages, **(model_kwargs or {})):
                if response is None:
                    response = delta
                else:
                    response += delta

                yield AgentMessageDelta(agent=self, delta=delta, snapshot=response)

        else:
            response: AIMessage = model.invoke(messages)

        yield AgentMessage(agent=self, message=response)
        create_markdown_artifact(
            markdown=f"""
{response.content or '(No content)'}

#### Payload
```json
{response.model_dump_json(indent=2)}
```
""",
            description=f"LLM Response for Agent {self.name}",
            key="agent-message",
        )

        if controlflow.settings.log_all_messages:
            logger.debug(f"Response: {response}")

        for tool_call in response.tool_calls + response.invalid_tool_calls:
            yield ToolCallEvent(agent=self, tool_call=tool_call)
            result = handle_tool_call(tool_call, tools=tools)
            yield ToolResultEvent(agent=self, tool_call=tool_call, tool_result=result)

    @prefect_task(task_run_name="Call LLM")
    async def _run_model_async(
        self,
        messages: list[BaseMessage],
        tools: list["Tool"],
        stream: bool = True,
        model_kwargs: Optional[dict] = None,
    ) -> AsyncGenerator[Event, None]:
        from controlflow.events.events import (
            AgentMessage,
            AgentMessageDelta,
            ToolCallEvent,
            ToolResultEvent,
        )

        tools = as_tools(self.get_tools() + tools)
        model = self.get_model(tools=tools)

        logger.debug(f"Running model {model} for agent {self.name} with tools {tools}")
        if controlflow.settings.log_all_messages:
            logger.debug(f"Input messages: {messages}")

        if stream:
            response = None
            async for delta in model.astream(messages, **(model_kwargs or {})):
                if response is None:
                    response = delta
                else:
                    response += delta

                yield AgentMessageDelta(agent=self, delta=delta, snapshot=response)

        else:
            response: AIMessage = await model.ainvoke(messages)

        yield AgentMessage(agent=self, message=response)

        create_markdown_artifact(
            markdown=f"""
{response.content or '(No content)'}

#### Payload
```json
{response.model_dump_json(indent=2)}
```
""",
            description=f"LLM Response for Agent {self.name}",
            key="agent-message",
        )

        if controlflow.settings.log_all_messages:
            logger.debug(f"Response: {response}")

        for tool_call in response.tool_calls + response.invalid_tool_calls:
            yield ToolCallEvent(agent=self, tool_call=tool_call)
            result = await handle_tool_call_async(tool_call, tools=tools)
            yield ToolResultEvent(agent=self, tool_call=tool_call, tool_result=result)
