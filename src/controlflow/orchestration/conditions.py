import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from pydantic import BaseModel, field_validator

from controlflow.tasks.task import Task
from controlflow.utilities.general import ControlFlowModel
from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.orchestration.orchestrator import Orchestrator

logger = get_logger(__name__)


class RunContext(ControlFlowModel):
    """
    Context for a run.
    """

    model_config = dict(arbitrary_types_allowed=True)

    orchestrator: "Orchestrator"
    llm_calls: int = 0
    agent_turns: int = 0
    run_end_condition: "RunEndCondition"

    @field_validator("run_end_condition", mode="before")
    def validate_condition(cls, v: Any) -> "RunEndCondition":
        if not isinstance(v, RunEndCondition):
            v = FnCondition(v)
        return v

    def should_end(self) -> bool:
        return self.run_end_condition.should_end(self)


class RunEndCondition:
    def should_end(self, context: RunContext) -> bool:
        """
        Returns True if the run should end, False otherwise.
        """
        return False

    def __or__(
        self, other: Union["RunEndCondition", Callable[[RunContext], bool]]
    ) -> "RunEndCondition":
        if isinstance(other, RunEndCondition):
            return OR_(self, other)
        elif callable(other):
            return OR_(self, FnCondition(other))
        else:
            raise NotImplementedError(
                f"Cannot combine RunEndCondition with {type(other)}"
            )

    def __and__(
        self, other: Union["RunEndCondition", Callable[[RunContext], bool]]
    ) -> "RunEndCondition":
        if isinstance(other, RunEndCondition):
            return AND_(self, other)
        elif callable(other):
            return AND_(self, FnCondition(other))
        else:
            raise NotImplementedError(
                f"Cannot combine RunEndCondition with {type(other)}"
            )


class FnCondition(RunEndCondition):
    def __init__(self, fn: Callable[[RunContext], bool]):
        self.fn = fn

    def should_end(self, context: RunContext) -> bool:
        result = self.fn(context)
        if result:
            logger.debug("Custom function condition met; ending run.")
        return result


class OR_(RunEndCondition):
    def __init__(self, *conditions: RunEndCondition):
        self.conditions = conditions

    def should_end(self, context: RunContext) -> bool:
        result = any(condition.should_end(context) for condition in self.conditions)
        if result:
            logger.debug("At least one condition in OR clause met.")
        return result


class AND_(RunEndCondition):
    def __init__(self, *conditions: RunEndCondition):
        self.conditions = conditions

    def should_end(self, context: RunContext) -> bool:
        result = all(condition.should_end(context) for condition in self.conditions)
        if result:
            logger.debug("All conditions in AND clause met.")
        return result


class AllComplete(RunEndCondition):
    def __init__(self, tasks: Optional[list[Task]] = None):
        self.tasks = tasks

    def should_end(self, context: RunContext) -> bool:
        tasks = self.tasks if self.tasks is not None else context.orchestrator.tasks
        result = all(t.is_complete() for t in tasks)
        if result:
            logger.debug("All tasks are complete; ending run.")
        return result


class AnyComplete(RunEndCondition):
    def __init__(self, tasks: Optional[list[Task]] = None, min_complete: int = 1):
        self.tasks = tasks
        if min_complete < 1:
            raise ValueError("min_complete must be at least 1")
        self.min_complete = min_complete

    def should_end(self, context: RunContext) -> bool:
        tasks = self.tasks if self.tasks is not None else context.orchestrator.tasks
        result = sum(t.is_complete() for t in tasks) >= self.min_complete
        if result:
            logger.debug("At least one task is complete; ending run.")
        return result


class AnyFailed(RunEndCondition):
    def __init__(self, tasks: Optional[list[Task]] = None, min_failed: int = 1):
        self.tasks = tasks
        if min_failed < 1:
            raise ValueError("min_failed must be at least 1")
        self.min_failed = min_failed

    def should_end(self, context: RunContext) -> bool:
        tasks = self.tasks if self.tasks is not None else context.orchestrator.tasks
        result = sum(t.is_failed() for t in tasks) >= self.min_failed
        if result:
            logger.debug("At least one task has failed; ending run.")
        return result


class MaxAgentTurns(RunEndCondition):
    def __init__(self, n: int):
        self.n = n

    def should_end(self, context: RunContext) -> bool:
        result = context.agent_turns >= self.n
        if result:
            logger.debug(
                f"Maximum number of agent turns ({self.n}) reached; ending run."
            )
        return result


class MaxLLMCalls(RunEndCondition):
    def __init__(self, n: int):
        self.n = n

    def should_end(self, context: RunContext) -> bool:
        result = context.llm_calls >= self.n
        if result:
            logger.debug(f"Maximum number of LLM calls ({self.n}) reached; ending run.")
        return result
