import logging
from typing import Callable, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator

import controlflow
from controlflow.agents.agent import Agent
from controlflow.events.base import Event
from controlflow.events.events import AgentMessageDelta, OrchestratorMessage
from controlflow.events.message_compiler import MessageCompiler
from controlflow.flows import Flow
from controlflow.instructions import get_instructions
from controlflow.llm.messages import BaseMessage
from controlflow.memory import Memory
from controlflow.orchestration.conditions import (
    AllComplete,
    FnCondition,
    MaxAgentTurns,
    MaxLLMCalls,
    RunContext,
    RunEndCondition,
)
from controlflow.orchestration.handler import Handler
from controlflow.orchestration.turn_strategies import Popcorn, TurnStrategy
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, as_tools
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
    handlers: list[Handler] = Field(None, validate_default=True)

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
        from controlflow.orchestration.print_handler import PrintHandler

        if v is None and controlflow.settings.pretty_print_agent_events:
            v = [PrintHandler()]
        return v or []

    def handle_event(self, event: Event):
        """
        Handle an event by passing it to all handlers and persisting if necessary.

        Args:
            event (Event): The event to handle.
        """
        if not isinstance(event, AgentMessageDelta):
            logger.debug(f"Handling event: {repr(event)}")
        for handler in self.handlers:
            handler.handle(event)
        if event.persist:
            self.flow.add_events([event])

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

    def get_memories(self) -> list[Memory]:
        memories = set()

        memories.update(self.agent.memories)

        for task in self.get_tasks("assigned"):
            memories.update(task.memories)

        return memories

    @prefect_task(task_run_name="Orchestrator.run()")
    def run(
        self,
        max_llm_calls: Optional[int] = None,
        max_agent_turns: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        run_until: Optional[
            Union[RunEndCondition, Callable[[RunContext], bool]]
        ] = None,
    ) -> RunContext:
        import controlflow.events.orchestrator_events

        # Create the base termination condition
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

        # Initialize the agent if not already set
        if not self.agent:
            self.agent = self.turn_strategy.get_next_agent(
                None, self.get_available_agents()
            )

        # Signal the start of orchestration
        self.handle_event(
            controlflow.events.orchestrator_events.OrchestratorStart(orchestrator=self)
        )

        try:
            while True:
                if run_context.should_end():
                    break

                self.handle_event(
                    controlflow.events.orchestrator_events.AgentTurnStart(
                        orchestrator=self, agent=self.agent
                    )
                )
                self.run_agent_turn(
                    run_context=run_context,
                    model_kwargs=model_kwargs,
                )
                self.handle_event(
                    controlflow.events.orchestrator_events.AgentTurnEnd(
                        orchestrator=self, agent=self.agent
                    )
                )

                # Select the next agent for the following turn
                if available_agents := self.get_available_agents():
                    self.agent = self.turn_strategy.get_next_agent(
                        self.agent, available_agents
                    )

        except Exception as exc:
            # Handle any exceptions that occur during orchestration
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorError(
                    orchestrator=self, error=exc
                )
            )
            raise
        finally:
            # Signal the end of orchestration
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorEnd(
                    orchestrator=self
                )
            )
        return run_context

    @prefect_task
    async def run_async(
        self,
        max_llm_calls: Optional[int] = None,
        max_agent_turns: Optional[int] = None,
        model_kwargs: Optional[dict] = None,
        run_until: Optional[
            Union[RunEndCondition, Callable[[RunContext], bool]]
        ] = None,
    ) -> RunContext:
        import controlflow.events.orchestrator_events

        # Create the base termination condition
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

        # Initialize the agent if not already set
        if not self.agent:
            self.agent = self.turn_strategy.get_next_agent(
                None, self.get_available_agents()
            )

        # Signal the start of orchestration
        self.handle_event(
            controlflow.events.orchestrator_events.OrchestratorStart(orchestrator=self)
        )

        try:
            while True:
                # Check termination condition
                if run_context.should_end():
                    break

                self.handle_event(
                    controlflow.events.orchestrator_events.AgentTurnStart(
                        orchestrator=self, agent=self.agent
                    )
                )
                await self.run_agent_turn_async(
                    run_context=run_context,
                    model_kwargs=model_kwargs,
                )
                self.handle_event(
                    controlflow.events.orchestrator_events.AgentTurnEnd(
                        orchestrator=self, agent=self.agent
                    )
                )

                # Select the next agent for the following turn
                if available_agents := self.get_available_agents():
                    self.agent = self.turn_strategy.get_next_agent(
                        self.agent, available_agents
                    )

        except Exception as exc:
            # Handle any exceptions that occur during orchestration
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorError(
                    orchestrator=self, error=exc
                )
            )
            raise
        finally:
            # Signal the end of orchestration
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorEnd(
                    orchestrator=self
                )
            )
        return run_context

    @prefect_task(task_run_name="Agent turn: {self.agent.name}")
    def run_agent_turn(
        self,
        run_context: RunContext,
        model_kwargs: Optional[dict] = None,
    ) -> int:
        """
        Run a single agent turn, which may consist of multiple LLM calls.
        """
        assigned_tasks = self.get_tasks("assigned")

        self.turn_strategy.begin_turn()

        # Mark assigned tasks as running
        for task in assigned_tasks:
            if not task.is_running():
                task.mark_running()
                self.handle_event(
                    OrchestratorMessage(
                        content=f"Starting task {task.name + ' ' if task.name else ''}(ID {task.id}) "
                        f"with objective: {task.objective}"
                    )
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

            for event in self.agent._run_model(
                messages=messages,
                tools=tools,
                model_kwargs=model_kwargs,
            ):
                self.handle_event(event)

            run_context.llm_calls += 1
            for task in assigned_tasks:
                task._llm_calls += 1

        run_context.agent_turns += 1

    @prefect_task
    async def run_agent_turn_async(
        self,
        run_context: RunContext,
        model_kwargs: Optional[dict] = None,
    ) -> int:
        """
        Run a single agent turn asynchronously, which may consist of multiple LLM calls.

        Args:
            max_llm_calls (Optional[int]): The number of LLM calls allowed.

        Returns:
            int: The number of LLM calls made during this turn.
        """
        assigned_tasks = self.get_tasks("assigned")

        self.turn_strategy.begin_turn()

        # Mark assigned tasks as running
        for task in assigned_tasks:
            if not task.is_running():
                task.mark_running()
                self.handle_event(
                    OrchestratorMessage(
                        content=f"Starting task {task.name} (ID {task.id}) "
                        f"with objective: {task.objective}"
                    )
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

            async for event in self.agent._run_model_async(
                messages=messages,
                tools=tools,
                model_kwargs=model_kwargs,
            ):
                self.handle_event(event)

            run_context.llm_calls += 1
            for task in assigned_tasks:
                task._llm_calls += 1

        run_context.agent_turns += 1

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


RunContext.model_rebuild()
