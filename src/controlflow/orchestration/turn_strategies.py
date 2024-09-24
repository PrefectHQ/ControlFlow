import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from controlflow.agents import Agent
from controlflow.tasks.task import Task
from controlflow.tools.tools import Tool, tool
from controlflow.utilities.general import ControlFlowModel


class TurnStrategy(ControlFlowModel, ABC):
    end_turn: bool = False
    next_agent: Optional[Agent] = None

    @abstractmethod
    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        pass

    @abstractmethod
    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        pass

    def begin_turn(self):
        self.end_turn = False
        self.next_agent = None

    def should_end_turn(self) -> bool:
        """
        Determine if the current turn should end.

        Returns:
            bool: True if the turn should end, False otherwise.
        """
        return self.end_turn


def get_end_turn_tool(strategy: TurnStrategy) -> Tool:
    @tool
    def end_turn() -> str:
        """
        End your turn. Only use this tool if you have no other options and
        want a different agent to take over. This tool does not mark tasks as complete.
        """
        strategy.end_turn = True
        return "Turn ended."

    return end_turn


def get_delegate_tool(
    strategy: TurnStrategy, available_agents: dict[Agent, list[Task]]
) -> Tool:
    @tool
    def delegate_to_agent(agent_id: str, message: str = None) -> str:
        """Delegate to another agent and optionally send a message."""
        if len(available_agents) <= 1:
            return "Cannot delegate as there are no other available agents."
        next_agent = next(
            (a for a in available_agents.keys() if a.id == agent_id), None
        )
        if next_agent is None:
            raise ValueError(f"Agent with ID {agent_id} not found or not available.")
        strategy.end_turn = True
        strategy.next_agent = next_agent
        return f"Delegated to agent {next_agent.name} with ID {agent_id}"

    return delegate_to_agent


class SingleAgent(TurnStrategy):
    agent: Agent

    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        return [get_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        if self.agent not in available_agents:
            raise ValueError(
                "The agent specified by the turn strategy is not available."
            )
        return self.agent


class Popcorn(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        return [get_delegate_tool(self, available_agents)]

    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        if self.next_agent and self.next_agent in available_agents:
            return self.next_agent
        return next(iter(available_agents))


class Random(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        return [get_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        return random.choice(list(available_agents.keys()))


class RoundRobin(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        return [get_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        agents = list(available_agents.keys())
        if current_agent is None or current_agent not in agents:
            return agents[0]
        current_index = agents.index(current_agent)
        next_index = (current_index + 1) % len(agents)
        return agents[next_index]


class MostBusy(TurnStrategy):
    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        return [get_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        # Select the agent with the most tasks
        return max(available_agents, key=lambda agent: len(available_agents[agent]))


class Moderated(TurnStrategy):
    moderator: Agent

    def get_tools(
        self, current_agent: Agent, available_agents: dict[Agent, list[Task]]
    ) -> list[Tool]:
        if current_agent == self.moderator:
            return [get_delegate_tool(self, available_agents)]
        else:
            return [get_end_turn_tool(self)]

    def get_next_agent(
        self, current_agent: Optional[Agent], available_agents: Dict[Agent, List[Task]]
    ) -> Agent:
        if current_agent is None or current_agent is self.moderator:
            return (
                self.next_agent
                if self.next_agent in available_agents
                else self.moderator
            )
        else:
            return self.moderator
