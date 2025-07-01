import logging
from collections import deque
from contextlib import contextmanager
from typing import AsyncIterator, Callable, Iterator, Optional, Set, TypeVar, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator

import controlflow
from controlflow.agents.agent import Agent
from controlflow.events.base import Event
from controlflow.events.events import AgentMessageDelta, OrchestratorMessage
from controlflow.events.message_compiler import MessageCompiler
from controlflow.events.orchestrator_events import (
    AgentTurnEnd,
    AgentTurnStart,
    OrchestratorEnd,
    OrchestratorError,
    OrchestratorStart,
)
from controlflow.flows import Flow
from controlflow.instructions import get_instructions
from controlflow.llm.messages import BaseMessage
from controlflow.memory import Memory
from controlflow.memory.async_memory import AsyncMemory
from controlflow.orchestration.conditions import (
    AllComplete,
    FnCondition,
    MaxAgentTurns,
    MaxLLMCalls,
    RunContext,
    RunEndCondition,
)
from controlflow.orchestration.handler import AsyncHandler, Handler
from controlflow.orchestration.turn_strategies import Popcorn, TurnStrategy
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, as_tools
from controlflow.utilities.context import ctx
from controlflow.utilities.general import ControlFlowModel
from controlflow.utilities.prefect import prefect_task

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Orchestrator(ControlFlowModel):
    """
    The orchestrator is responsible for managing the flow of tasks and agents.
    It is given tasks to execute in a flow context, and an agent to execute the
    tasks. The turn strategy determines how agents take turns and collaborate.
    """

    model_config = dict(arbitrary_types_allowed=True)
    flow: "Flow" = Field(description="The flow that the orchestrator is managing")
    agent: Optional[Agent] = Field(
        None,
        description="The currently active agent. If not provided, the turn strategy will select one.",
    )
    tasks: list[Task] = Field(description="Tasks to be executed by the agent.")
    turn_strategy: TurnStrategy = Field(
        default=None,
        description="The strategy to use for managing agent turns",
        validate_default=True,
    )
    handlers: list[Union[Handler, AsyncHandler]] = Field(
        None, validate_default=True, exclude=True
    )
    _processed_event_ids: Set[str] = PrivateAttr(default_factory=set)
    _pending_events: deque[Event] = PrivateAttr(default_factory=deque)

    @field_validator("turn_strategy", mode="before")
    def _validate_turn_strategy(cls, v):
        if v is None:
            v = Popcorn()
        return v

    @field_validator("handlers", mode="before")
    def _validate_handlers(cls, v):
        """
        Validate and set default handlers.

        Args:
            v: The input value for handlers.

        Returns:
            list[Handler]: The validated list of handlers.
        """
        from controlflow.handlers.print_handler import PrintHandler

        if v is None and controlflow.settings.enable_default_print_handler:
            v = [
                PrintHandler(
                    show_completion_tools=controlflow.settings.default_print_handler_show_completion_tools,
                    show_completion_tool_results=controlflow.settings.default_print_handler_show_completion_tool_results,
                )
            ]
        return v or []

    def add_event(self, event: Event) -> None:
        """Add an event to be handled and yielded during the next run loop iteration"""
        self._pending_events.append(event)

    def handle_event(self, event: Event) -> Event:
        """
        Handle an event by passing it to all handlers and persisting if necessary.
        Includes idempotency check to prevent double-processing events.
        """
        # Skip if we've already processed this event
        if event.id in self._processed_event_ids:
            return event

        for handler in self.handlers:
            if isinstance(handler, Handler):
                handler.handle(event)
        if event.persist:
            self.flow.add_events([event])

        # Mark event as processed
        self._processed_event_ids.add(event.id)
        return event

    async def handle_event_async(self, event: Event) -> Event:
        """
        Handle an event asynchronously by passing it to all handlers and persisting if necessary.
        Includes idempotency check to prevent double-processing events.

        Args:
            event (Event): The event to handle.
        """
        # Skip if we've already processed this event
        if event.id in self._processed_event_ids:
            return event

        if not isinstance(event, AgentMessageDelta):
            logger.debug(f"Handling event asynchronously: {repr(event)}")
        for handler in self.handlers:
            if isinstance(handler, AsyncHandler):
                await handler.handle(event)
            elif isinstance(handler, Handler):
                handler.handle(event)
        if event.persist:
            self.flow.add_events([event])

        # Mark event as processed
        self._processed_event_ids.add(event.id)
        return event

    def get_available_agents(self) -> dict[Agent, list[Task]]:
        """
        Get a dictionary of all available agents for active tasks, mapped to
        their assigned tasks.

        Returns:
            dict[Agent, list[Task]]
        """
        ready_tasks = self.get_tasks("ready")
        agents = {}
        for task in ready_tasks:
            for agent in task.get_agents():
                agents.setdefault(agent, []).append(task)
        return agents

    def get_tools(self) -> list[Tool]:
        """
        Get all tools available for the current turn.

        Returns:
            list[Tool]: A list of available tools.
        """
        tools = []

        # add flow tools
        tools.extend(self.flow.tools)

        # add task tools
        for task in self.get_tasks("assigned"):
            tools.extend(task.get_tools())

            # add completion tools
            if task.completion_agents is None or self.agent in task.completion_agents:
                tools.extend(task.get_completion_tools())

        # add turn strategy tools only if there are multiple available agents
        available_agents = self.get_available_agents()
        if len(available_agents) > 1:
            tools.extend(self.turn_strategy.get_tools(self.agent, available_agents))

        tools = as_tools(tools)
        return tools

    def get_memories(self) -> list[Union[Memory, AsyncMemory]]:
        memories = set()

        memories.update(self.agent.memories)

        for task in self.get_tasks("assigned"):
            memories.update(task.memories)

        return memories

    def _run_agent_turn(
        self,
        run_context: RunContext,
        model_kwargs: Optional[dict] = None,
    ) -> Iterator[Event]:
        """Run a single agent turn, yielding events as they occur."""
        assigned_tasks = self.get_tasks("assigned")

        self.turn_strategy.begin_turn()

        # Mark assigned tasks as running
        for task in assigned_tasks:
            if not task.is_running():
                task.mark_running()
                yield OrchestratorMessage(
                    content=f"Starting task {task.name + ' ' if task.name else ''}(ID {task.id}) "
                    f"with objective: {task.objective}"
                )

        while not self.turn_strategy.should_end_turn():
            # fail any tasks that have reached their max llm calls
            for task in assigned_tasks:
                if task.max_llm_calls and task._llm_calls >= task.max_llm_calls:
                    task.mark_failed(reason="Max LLM calls reached for this task.")

            # Check if there are any ready tasks left
            if not any(t.is_ready() for t in assigned_tasks):
                logger.debug("No `ready` tasks to run")
                break

            if run_context.should_end():
                break

            messages = self.compile_messages()
            tools = self.get_tools()

            # Run model and yield events
            with ctx(orchestrator=self):
                for event in self.agent._run_model(
                    messages=messages,
                    tools=tools,
                    model_kwargs=model_kwargs,
                ):
                    yield event

            run_context.llm_calls += 1
            for task in assigned_tasks:
                task._llm_calls += 1

        run_context.agent_turns += 1

    @prefect_task(task_run_name="Run agent orchestrator")
    def _run(
        self,
        run_context: RunContext,
        model_kwargs: Optional[dict] = None,
    ) -> Iterator[Event]:
        """Run the orchestrator, yielding handled events as they occur."""
        # Initialize the agent if not already set
        if not self.agent:
            self.agent = self.turn_strategy.get_next_agent(
                None, self.get_available_agents()
            )

        # Signal the start of orchestration
        start_event = OrchestratorStart(orchestrator=self, run_context=run_context)
        self.handle_event(start_event)
        yield start_event

        try:
            while True:
                if run_context.should_end():
                    break

                turn_start = AgentTurnStart(orchestrator=self, agent=self.agent)
                self.handle_event(turn_start)
                yield turn_start

                # Run turn and yield its events
                for event in self._run_agent_turn(
                    run_context=run_context,
                    model_kwargs=model_kwargs,
                ):
                    self.handle_event(event)
                    yield event

                # Handle any events added during the turn
                while self._pending_events:
                    event = self._pending_events.popleft()
                    self.handle_event(event)
                    yield event

                turn_end = AgentTurnEnd(orchestrator=self, agent=self.agent)
                self.handle_event(turn_end)
                yield turn_end

                # Select the next agent for the following turn
                if available_agents := self.get_available_agents():
                    self.agent = self.turn_strategy.get_next_agent(
                        self.agent, available_agents
                    )

        except Exception as exc:
            # Yield error event if something goes wrong
            error_event = OrchestratorError(orchestrator=self, error=exc)
            self.handle_event(error_event)
            yield error_event
            raise
        finally:
            # Signal the end of orchestration
            end_event = OrchestratorEnd(orchestrator=self, run_context=run_context)
            self.handle_event(end_event)
            yield end_event

            # Handle any final pending events
            while self._pending_events:
                event = self._pending_events.popleft()
                self.handle_event(event)
                yield event

    def run(
        self,
        max_llm_calls: Optional[int] = None,
        max_agent_turns: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        run_until: Optional[
            Union[RunEndCondition, Callable[[RunContext], bool]]
        ] = None,
        stream: bool = False,
    ) -> Union[RunContext, Iterator[Event]]:
        """
        Run the orchestrator.

        Args:
            max_llm_calls: Maximum number of LLM calls allowed
            max_agent_turns: Maximum number of agent turns allowed
            model_kwargs: Additional kwargs for the model
            run_until: Condition for ending the run
            stream: If True, return iterator of events. If False, consume events and return context

        Returns:
            If stream=True, returns Iterator[Event]
            If stream=False, returns RunContext
        """
        # Create run context at the outermost level
        if run_until is None:
            run_until = AllComplete()
        elif not isinstance(run_until, RunEndCondition):
            run_until = FnCondition(run_until)

        # Add max_llm_calls condition
        if max_llm_calls is None:
            max_llm_calls = controlflow.settings.orchestrator_max_llm_calls
        run_until = run_until | MaxLLMCalls(max_llm_calls)

        # Add max_agent_turns condition
        if max_agent_turns is None:
            max_agent_turns = controlflow.settings.orchestrator_max_agent_turns
        run_until = run_until | MaxAgentTurns(max_agent_turns)

        run_context = RunContext(orchestrator=self, run_end_condition=run_until)

        result = self._run(
            run_context=run_context,
            model_kwargs=model_kwargs,
        )

        if stream:
            return result
        else:
            for _ in result:
                pass
            return run_context

    @prefect_task(task_run_name="Run agent orchestrator")
    async def _run_async(
        self,
        run_context: RunContext,
        model_kwargs: Optional[dict] = None,
    ) -> AsyncIterator[Event]:
        """Run the orchestrator asynchronously, yielding handled events as they occur."""
        # Initialize the agent if not already set
        if not self.agent:
            self.agent = self.turn_strategy.get_next_agent(
                None, self.get_available_agents()
            )

        # Signal the start of orchestration
        start_event = OrchestratorStart(orchestrator=self, run_context=run_context)
        await self.handle_event_async(start_event)
        yield start_event

        try:
            while True:
                if run_context.should_end():
                    break

                turn_start = AgentTurnStart(orchestrator=self, agent=self.agent)
                await self.handle_event_async(turn_start)
                yield turn_start

                # Run turn and yield its events
                async for event in self._run_agent_turn_async(
                    run_context=run_context,
                    model_kwargs=model_kwargs,
                ):
                    await self.handle_event_async(event)
                    yield event

                # Handle any events added during the turn
                while self._pending_events:
                    event = self._pending_events.popleft()
                    await self.handle_event_async(event)
                    yield event

                turn_end = AgentTurnEnd(orchestrator=self, agent=self.agent)
                await self.handle_event_async(turn_end)
                yield turn_end

                # Select the next agent for the following turn
                if available_agents := self.get_available_agents():
                    self.agent = self.turn_strategy.get_next_agent(
                        self.agent, available_agents
                    )

        except Exception as exc:
            # Yield error event if something goes wrong
            error_event = OrchestratorError(orchestrator=self, error=exc)
            await self.handle_event_async(error_event)
            yield error_event
            raise
        finally:
            # Signal the end of orchestration
            end_event = OrchestratorEnd(orchestrator=self, run_context=run_context)
            await self.handle_event_async(end_event)
            yield end_event

            # Handle any final pending events
            while self._pending_events:
                event = self._pending_events.popleft()
                await self.handle_event_async(event)
                yield event

    async def run_async(
        self,
        max_llm_calls: Optional[int] = None,
        max_agent_turns: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        run_until: Optional[
            Union[RunEndCondition, Callable[[RunContext], bool]]
        ] = None,
        stream: bool = False,
    ) -> Union[RunContext, AsyncIterator[Event]]:
        """
        Run the orchestrator asynchronously.

        Args:
            max_llm_calls: Maximum number of LLM calls allowed
            max_agent_turns: Maximum number of agent turns allowed
            model_kwargs: Additional kwargs for the model
            run_until: Condition for ending the run
            stream: If True, return async iterator of events. If False, consume events and return context

        Returns:
            If stream=True, returns AsyncIterator[Event]
            If stream=False, returns RunContext
        """
        # Create run context at the outermost level
        if run_until is None:
            run_until = AllComplete()
        elif not isinstance(run_until, RunEndCondition):
            run_until = FnCondition(run_until)

        # Add max_llm_calls condition
        if max_llm_calls is None:
            max_llm_calls = controlflow.settings.orchestrator_max_llm_calls
        run_until = run_until | MaxLLMCalls(max_llm_calls)

        # Add max_agent_turns condition
        if max_agent_turns is None:
            max_agent_turns = controlflow.settings.orchestrator_max_agent_turns
        run_until = run_until | MaxAgentTurns(max_agent_turns)

        run_context = RunContext(orchestrator=self, run_end_condition=run_until)

        result = self._run_async(
            run_context=run_context,
            model_kwargs=model_kwargs,
        )

        if stream:
            return result
        else:
            async for _ in result:
                pass
            return run_context

    def compile_prompt(self) -> str:
        """
        Compile the prompt for the current turn.

        Returns:
            str: The compiled prompt.
        """
        from controlflow.orchestration.prompt_templates import (
            InstructionsTemplate,
            LLMInstructionsTemplate,
            MemoryTemplate,
            TasksTemplate,
            ToolTemplate,
        )

        llm_rules = self.agent.get_llm_rules()

        prompts = [
            self.agent.get_prompt(),
            self.flow.get_prompt(),
            TasksTemplate(tasks=self.get_tasks("ready")).render(),
            ToolTemplate(tools=self.get_tools()).render(),
            MemoryTemplate(memories=self.get_memories()).render(),
            InstructionsTemplate(instructions=get_instructions()).render(),
            LLMInstructionsTemplate(
                instructions=llm_rules.model_instructions()
            ).render(),
        ]

        prompt = "\n\n".join([p for p in prompts if p])
        logger.debug(f"{'=' * 10}\nCompiled prompt: {prompt}\n{'=' * 10}")
        return prompt

    def compile_messages(self) -> list[BaseMessage]:
        """
        Compile messages for the current turn.

        Returns:
            list[BaseMessage]: The compiled messages.
        """
        events = self.flow.get_events(limit=100)

        compiler = MessageCompiler(
            events=events,
            llm_rules=self.agent.get_llm_rules(),
            system_prompt=self.compile_prompt(),
        )
        messages = compiler.compile_to_messages(agent=self.agent)
        return messages

    def get_tasks(self, filter: str = "assigned") -> list[Task]:
        """
        Collect tasks based on the specified filter.

        Args:
            filter (str): Determines which tasks to return.
                - "ready": Tasks ready to execute (no unmet dependencies).
                - "assigned": Ready tasks assigned to the current agent.
                - "all": All tasks including subtasks, dependencies, and direct ancestors of root tasks.

        Returns:
            list[Task]: List of tasks based on the specified filter.
        """
        if filter not in ["ready", "assigned", "all"]:
            raise ValueError(f"Invalid filter: {filter}")

        all_tasks: set[Task] = set()
        ready_tasks: list[Task] = []

        def collect_tasks(task: Task):
            if task in all_tasks:
                return
            all_tasks.add(task)

            # Collect subtasks
            for subtask in task.subtasks:
                collect_tasks(subtask)

            # Collect dependencies
            for dependency in task.depends_on:
                collect_tasks(dependency)

            # Collect parent
            if task.parent and not task.parent.wait_for_subtasks:
                collect_tasks(task.parent)

            # Check if the task is ready
            if task.is_ready():
                ready_tasks.append(task)

        # Collect tasks from self.tasks (root tasks) and their direct ancestors
        for task in self.tasks:
            collect_tasks(task)

            # Collect direct ancestors of root tasks
            current = task.parent
            while current:
                all_tasks.add(current)
                current = current.parent

        if filter == "ready":
            return ready_tasks

        if filter == "assigned":
            return [task for task in ready_tasks if self.agent in task.get_agents()]

        # "all" filter
        return list(all_tasks)

    def get_task_hierarchy(self) -> dict:
        """
        Build a hierarchical structure of all tasks.

        Returns:
            dict: A nested dictionary representing the task hierarchy,
            where each task has 'task' and 'children' keys.
        """
        all_tasks = self.get_tasks("all")

        hierarchy = {}
        task_dict_map = {task.id: {"task": task, "children": []} for task in all_tasks}

        for task in all_tasks:
            if task.parent:
                parent_dict = task_dict_map[task.parent.id]
                parent_dict["children"].append(task_dict_map[task.id])
            else:
                hierarchy[task.id] = task_dict_map[task.id]

        return hierarchy

    async def _run_agent_turn_async(
        self,
        run_context: RunContext,
        model_kwargs: Optional[dict] = None,
    ) -> AsyncIterator[Event]:
        """Async version of _run_agent_turn."""
        assigned_tasks = self.get_tasks("assigned")

        self.turn_strategy.begin_turn()

        # Mark assigned tasks as running
        for task in assigned_tasks:
            if not task.is_running():
                task.mark_running()
                yield OrchestratorMessage(
                    content=f"Starting task {task.name} (ID {task.id}) "
                    f"with objective: {task.objective}"
                )

        while not self.turn_strategy.should_end_turn():
            # fail any tasks that have reached their max llm calls
            for task in assigned_tasks:
                if task.max_llm_calls and task._llm_calls >= task.max_llm_calls:
                    task.mark_failed(reason="Max LLM calls reached for this task.")

            # Check if there are any ready tasks left
            if not any(t.is_ready() for t in assigned_tasks):
                logger.debug("No `ready` tasks to run")
                break

            if run_context.should_end():
                break

            messages = self.compile_messages()
            tools = self.get_tools()

            # Run model and yield events
            with ctx(orchestrator=self):
                async for event in self.agent._run_model_async(
                    messages=messages,
                    tools=tools,
                    model_kwargs=model_kwargs,
                ):
                    yield event

            run_context.llm_calls += 1
            for task in assigned_tasks:
                task._llm_calls += 1

        run_context.agent_turns += 1


# Rebuild all models with forward references after Orchestrator is defined
OrchestratorStart.model_rebuild()
OrchestratorEnd.model_rebuild()
OrchestratorError.model_rebuild()
AgentTurnStart.model_rebuild()
AgentTurnEnd.model_rebuild()
RunContext.model_rebuild()
