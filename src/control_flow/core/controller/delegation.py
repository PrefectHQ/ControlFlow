import itertools
from typing import TYPE_CHECKING, Any, Generator, Iterator

from pydantic import BaseModel, PrivateAttr

from control_flow.core.agent import Agent

if TYPE_CHECKING:
    from control_flow.core.agent import Agent


class DelegationStrategy(BaseModel):
    """
    A DelegationStrategy is a strategy for delegating tasks to AI assistants.
    """

    def __call__(self, agents: list["Agent"]) -> Agent:
        """
        Run the delegation strategy with a list of agents.
        """
        return self._next_agent(agents)

    def _next_agent(self, agents: list["Agent"], **kwargs) -> "Agent":
        """
        Select an agent from a list of agents.
        """
        raise NotImplementedError()


class Single(DelegationStrategy):
    """
    A Single delegation strategy delegates tasks to a single agent. This is useful for debugging.
    """

    agent: Agent

    def _next_agent(self, agents: list[Agent], **kwargs) -> Generator[Any, Any, Agent]:
        """
        Given a list of potential agents, choose the single agent.
        """
        if self.agent in agents:
            return self.agent


class RoundRobin(DelegationStrategy):
    """
    A RoundRobin delegation strategy delegates tasks to AI assistants in a round-robin fashion.
    """

    _cycle: Iterator[Agent] = PrivateAttr(None)

    def _next_agent(self, agents: list["Agent"], **kwargs) -> "Agent":
        """
        Given a list of potential agents, delegate the tasks in a round-robin fashion.
        """
        # the first time this is called, create a cycle iterator
        if self._cycle is None:
            self._cycle = itertools.cycle(agents)
        return next(self._cycle)


# class Moderator(DelegationStrategy):
#     """
#     A Moderator delegation strategy delegates tasks to the most qualified AI assistant, using a Marvin classifier
#     """

#     model: str = None

#     def _next_agent(
#         self, agents: list["Agent"], tasks: list[Task], history: list[Message]
#     ) -> "Agent":
#         """
#         Given a list of potential agents, choose the most qualified assistant to complete the tasks.
#         """

#         instructions = get_instructions()

#         context = dict(tasks=tasks, messages=history, global_instructions=instructions)
#         agent = marvin.classify(
#             context,
#             [a for a in agents if a.status == AgentStatus.INCOMPLETE],
#             instructions="""
#             Given the conversation context, choose the AI agent most
#             qualified to take the next turn at completing the tasks. Take into
#             account the instructions, each agent's own instructions, and the
#             tools they have available.
#             """,
#             model_kwargs=dict(model=self.model),
#         )

#         return agent
