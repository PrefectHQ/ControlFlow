import random
from abc import ABC, abstractmethod
from typing import List, Optional

from controlflow.agents import Agent
from controlflow.tools.tools import Tool, tool
from controlflow.utilities.general import ControlFlowModel


class TurnStrategy(ControlFlowModel, ABC):
    end_turn: bool = False
    next_agent: Optional[Agent] = None

    @abstractmethod
    def get_tools(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> List[Tool]:
        pass

    @abstractmethod
    def get_next_agent(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> Agent:
        pass

    def begin_turn(self):
        self.end_turn = False
        self.next_agent = None

    def should_end_turn(self) -> bool:
        return self.end_turn

    def should_end_session(self) -> bool:
        """
        Determine if the session should end.

        Returns:
            bool: True if the session should end, False otherwise.
        """
        return False


def create_end_turn_tool(strategy: TurnStrategy) -> Tool:
    @tool
    def end_turn() -> str:
        """End your turn."""
        strategy.end_turn = True
        return "Turn ended. Another agent will be selected."

    return end_turn


def create_delegate_tool(strategy: TurnStrategy, available_agents: List[Agent]) -> Tool:
    @tool
    def delegate_to_agent(agent_id: str) -> str:
        """Delegate to another agent."""
        next_agent = next((a for a in available_agents if a.id == agent_id), None)
        if next_agent is None:
            raise ValueError(f"Agent with ID {agent_id} not found or not available.")
        strategy.end_turn = True
        strategy.next_agent = next_agent
        return f"Delegated to agent {next_agent.name} with ID {agent_id}"

    return delegate_to_agent


class Single(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> List[Tool]:
        return [create_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> Agent:
        return current_agent

    def should_end_session(self) -> bool:
        return self.end_turn


class Popcorn(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> List[Tool]:
        return [create_delegate_tool(self, available_agents)]

    def get_next_agent(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> Agent:
        if self.next_agent and self.next_agent in available_agents:
            return self.next_agent
        return (
            current_agent if current_agent in available_agents else available_agents[0]
        )


class Random(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> List[Tool]:
        return [create_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> Agent:
        return random.choice(available_agents)


class RoundRobin(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> List[Tool]:
        return [create_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> Agent:
        if current_agent not in available_agents:
            return available_agents[0]
        current_index = available_agents.index(current_agent)
        next_index = (current_index + 1) % len(available_agents)
        return available_agents[next_index]


class Moderated(TurnStrategy):
    moderator: Agent

    def get_tools(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> List[Tool]:
        if current_agent == self.moderator:
            return [create_delegate_tool(self, available_agents)]
        else:
            return [create_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Agent, available_agents: List[Agent]
    ) -> Agent:
        if current_agent is self.moderator:
            return (
                self.next_agent
                if self.next_agent in available_agents
                else self.moderator
            )
        else:
            return self.moderator
