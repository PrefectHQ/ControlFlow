import itertools
from typing import TYPE_CHECKING, Any, Generator, Iterator

import marvin
from pydantic import BaseModel, PrivateAttr

from control_flow.agent import Agent, AgentStatus
from control_flow.flow import get_flow_messages
from control_flow.instructions import get_instructions

if TYPE_CHECKING:
    from control_flow.agent import Agent


class DelegationStrategy(BaseModel):
    """
    A DelegationStrategy is a strategy for delegating tasks to AI assistants.
    """

    def __call__(self, agents: list["Agent"]) -> Agent:
        """
        Run the delegation strategy with a list of agents.
        """
        return self._next_agent(agents)

    def _next_agent(self, agents: list["Agent"]) -> "Agent":
        """
        Select an agent from a list of agents.
        """
        raise NotImplementedError()


class Single(DelegationStrategy):
    """
    A Single delegation strategy delegates tasks to a single agent. This is useful for debugging.
    """

    agent: Agent

    def _next_agent(self, agents: list[Agent]) -> Generator[Any, Any, Agent]:
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

    def _next_agent(self, agents: list["Agent"]) -> "Agent":
        """
        Given a list of potential agents, delegate the tasks in a round-robin fashion.
        """
        # the first time this is called, create a cycle iterator
        if self._cycle is None:
            self._cycle = itertools.cycle(agents)

        # cycle once through all agents, returning the first one that is in the list
        # if no agent is found after cycling through all agents, return None
        first_agent_seen = None
        for agent in self._cycle:
            if agent in agents:
                return agent
            elif agent == first_agent_seen:
                break

            # remember the first agent seen so we can avoid an infinite loop
            if first_agent_seen is None:
                first_agent_seen = agent


class Moderator(DelegationStrategy):
    """
    A Moderator delegation strategy delegates tasks to the most qualified AI assistant, using a Marvin classifier
    """

    model: str = None

    def _next_agent(self, agents: list["Agent"]) -> "Agent":
        """
        Given a list of potential agents, choose the most qualified assistant to complete the tasks.
        """

        instructions = get_instructions()
        history = get_flow_messages()

        context = dict(messages=history, global_instructions=instructions)
        agent = marvin.classify(
            context,
            [a for a in agents if a.status == AgentStatus.INCOMPLETE],
            instructions="""
            Given the conversation context, choose the AI agent most
            qualified to take the next turn at completing the tasks. Take into
            account the instructions, each agent's own instructions, and the
            tools they have available.
            """,
            model_kwargs=dict(model=self.model),
        )

        return agent
