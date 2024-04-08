from typing import TYPE_CHECKING

from pydantic import BaseModel

from control_flow.task import TaskStatus

if TYPE_CHECKING:
    from control_flow.agent import Agent


class TerminationStrategy(BaseModel):
    """
    A TerminationStrategy is a strategy for deciding when AI assistants have completed their tasks.
    """

    def run(self, agents: list["Agent"]) -> bool:
        """
        Given agents, determine whether they have completed their tasks.
        """

        raise NotImplementedError()


class AllFinished(TerminationStrategy):
    """
    An AllFinished termination strategy terminates when all agents have finished all of their tasks (either COMPLETED or FAILED).
    """

    def run(self, agents: list["Agent"]) -> bool:
        """
        Given agents, determine whether they have completed their tasks.
        """
        for agent in agents:
            if any(task.status == TaskStatus.PENDING for task in agent.tasks):
                return False
        return True
