import logging
from typing import Optional, TypeVar

from pydantic import Field, field_validator

import controlflow
from controlflow.agents.agent import Agent
from controlflow.events.base import Event
from controlflow.events.message_compiler import MessageCompiler
from controlflow.flows import Flow
from controlflow.instructions import get_instructions
from controlflow.llm.messages import BaseMessage
from controlflow.orchestration.handler import Handler
from controlflow.orchestration.turn_strategies import Popcorn, TurnStrategy
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, as_tools
from controlflow.utilities.general import ControlFlowModel

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

        if v is None and controlflow.settings.enable_print_handler:
            v = [PrintHandler()]
        return v or []

    def handle_event(self, event: Event):
        """
        Handle an event by passing it to all handlers and persisting if necessary.

        Args:
            event (Event): The event to handle.
        """
        logger.debug(f"Handling event: {event}")
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
                tools.append(task.create_success_tool())
                tools.append(task.create_fail_tool())

        # add turn strategy tools
        tools.extend(
            self.turn_strategy.get_tools(self.agent, self.get_available_agents())
        )

        tools = as_tools(tools)
        return tools

    def _run_turn(self, max_calls_per_turn: Optional[int] = None):
        """
        Run a single turn of the orchestration process.

        Args:
            max_calls_per_turn (int, optional): Maximum number of LLM calls to run per turn.
        """
        if not self.agent:
            raise ValueError("No agent set.")

        if max_calls_per_turn is None:
            max_calls_per_turn = controlflow.settings.orchestrator_max_calls_per_turn

        self.turn_strategy.begin_turn()

        for task in self.get_tasks("assigned"):
            task.mark_running()

        calls = 0
        while not self.turn_strategy.should_end_turn():
            if max_calls_per_turn is not None and calls >= max_calls_per_turn:
                break
            calls += 1

            # Check if there are any ready tasks left
            if not self.get_tasks("ready"):
                break

            messages = self.compile_messages()
            tools = self.get_tools()

            for event in self.agent._run_model(messages=messages, tools=tools):
                self.handle_event(event)

        # at the end of each turn, select the next agent
        if available_agents := self.get_available_agents():
            self.agent = self.turn_strategy.get_next_agent(self.agent, available_agents)

    async def _run_turn_async(self, max_calls_per_turn: Optional[int] = None):
        """
        Run a single turn of the orchestration process asynchronously.

        Args:
            max_calls_per_turn (int, optional): Maximum number of LLM calls to run per turn.
        """
        if not self.agent:
            raise ValueError("No agent set.")

        if max_calls_per_turn is None:
            max_calls_per_turn = controlflow.settings.orchestrator_max_calls_per_turn

        self.turn_strategy.begin_turn()

        for task in self.get_tasks("assigned"):
            task.mark_running()

        calls = 0
        while not self.turn_strategy.should_end_turn():
            if max_calls_per_turn is not None and calls >= max_calls_per_turn:
                break
            calls += 1

            # Check if there are any ready tasks left
            if not self.get_tasks("ready"):
                break

            messages = self.compile_messages()
            tools = self.get_tools()
            async for event in self.agent._run_model_async(
                messages=messages, tools=tools
            ):
                self.handle_event(event)

        # at the end of each turn, select the next agent
        if available_agents := self.get_available_agents():
            self.agent = self.turn_strategy.get_next_agent(self.agent, available_agents)

    def run(
        self, max_turns: Optional[int] = None, max_calls_per_turn: Optional[int] = None
    ):
        """
        Run the orchestration process until the session should end.

        Args:
            turns (int, optional): Maximum number of turns to run.
            max_calls_per_turn (int, optional): Maximum number of LLM calls per turn.
        """
        import controlflow.events.orchestrator_events

        if not self.agent:
            self.agent = self.turn_strategy.get_next_agent(
                None, self.get_available_agents()
            )

        if max_turns is None:
            max_turns = controlflow.settings.orchestrator_max_turns

        self.handle_event(
            controlflow.events.orchestrator_events.OrchestratorStart(orchestrator=self)
        )

        turn = 0
        try:
            while self.get_tasks("ready"):
                if max_turns is not None and turn >= max_turns:
                    break
                self._run_turn(max_calls_per_turn=max_calls_per_turn)
                turn += 1
        except Exception as exc:
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorError(
                    orchestrator=self, error=exc
                )
            )
            raise
        finally:
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorEnd(
                    orchestrator=self
                )
            )

    async def run_async(
        self, max_turns: Optional[int] = None, max_calls_per_turn: Optional[int] = None
    ):
        """
        Run the orchestration process asynchronously until the session should end.

        Args:
            turns (int, optional): Maximum number of turns to run.
            max_calls_per_turn (int, optional): Maximum number of LLM calls per turn.
        """
        import controlflow.events.orchestrator_events

        if not self.agent:
            self.agent = self.turn_strategy.get_next_agent(
                None, self.get_available_agents()
            )

        if max_turns is None:
            max_turns = controlflow.settings.orchestrator_max_turns

        self.handle_event(
            controlflow.events.orchestrator_events.OrchestratorStart(orchestrator=self)
        )

        turn = 0
        try:
            while self.get_tasks("ready"):
                if max_turns is not None and turn >= max_turns:
                    break
                await self._run_turn_async(max_calls_per_turn=max_calls_per_turn)
                turn += 1
        except Exception as exc:
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorError(
                    orchestrator=self, error=exc
                )
            )
            raise
        finally:
            self.handle_event(
                controlflow.events.orchestrator_events.OrchestratorEnd(
                    orchestrator=self
                )
            )

    def compile_prompt(self) -> str:
        """
        Compile the prompt for the current turn.

        Returns:
            str: The compiled prompt.
        """
        from controlflow.orchestration.prompt_templates import (
            InstructionsTemplate,
            TasksTemplate,
            ToolTemplate,
        )

        tools = self.get_tools()

        prompts = [
            self.agent.get_prompt(),
            self.flow.get_prompt(),
            TasksTemplate(tasks=self.get_tasks("ready")).render(),
            ToolTemplate(tools=tools).render(),
            InstructionsTemplate(instructions=get_instructions()).render(),
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
                - "all": All tasks including subtasks and ancestors.

        Returns:
            list[Task]: List of tasks based on the specified filter.
        """
        if filter not in ["ready", "assigned", "all"]:
            raise ValueError(f"Invalid filter: {filter}")

        all_tasks: list[Task] = []
        ready_tasks: list[Task] = []

        def collect_tasks(task: Task, is_root: bool = False):
            if task not in all_tasks:
                all_tasks.append(task)
                if is_root and task.is_ready():
                    ready_tasks.append(task)
                for subtask in task.subtasks:
                    collect_tasks(subtask, is_root=is_root)

        # Collect tasks from self.tasks (root tasks)
        for task in self.tasks:
            collect_tasks(task, is_root=True)

        if filter == "ready":
            return ready_tasks

        if filter == "assigned":
            return [task for task in ready_tasks if self.agent in task.get_agents()]

        # Collect ancestor tasks for "all" filter
        for task in self.tasks:
            current = task.parent
            while current:
                if current not in all_tasks:
                    all_tasks.append(current)
                current = current.parent

        return all_tasks

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
