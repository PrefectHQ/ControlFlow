import pytest

from controlflow.agents.agent import Agent
from controlflow.orchestration.turn_strategies import (
    Moderated,
    MostBusy,
    Popcorn,
    Random,
    RoundRobin,
    SingleAgent,
)
from controlflow.tasks.task import Task


@pytest.fixture
def agents():
    return [Agent(name=f"Agent{i}") for i in range(1, 4)]


@pytest.fixture
def tasks(agents: list[Agent]):
    return [
        Task(objective=f"Task{i}", agents=[agents[i % len(agents)]]) for i in range(6)
    ]


@pytest.fixture
def available_agents(agents: list[Agent], tasks: list[Task]):
    return {
        agents[0]: tasks[0:3],
        agents[1]: tasks[3:5],
        agents[2]: tasks[5:6],
    }


def test_single_strategy(agents, available_agents):
    strategy = SingleAgent(agent=agents[0])
    current_agent = agents[0]

    tools = strategy.get_tools(current_agent, available_agents)
    assert len(tools) == 1
    assert tools[0].name == "end_turn"

    next_agent = strategy.get_next_agent(current_agent, available_agents)
    assert next_agent == current_agent


def test_popcorn_strategy(agents, available_agents):
    strategy = Popcorn()
    current_agent = agents[0]

    tools = strategy.get_tools(current_agent, available_agents)
    assert len(tools) == 1
    assert tools[0].name == "delegate_to_agent"

    next_agent = strategy.get_next_agent(current_agent, available_agents)
    assert next_agent == current_agent

    strategy.next_agent = agents[1]
    next_agent = strategy.get_next_agent(current_agent, available_agents)
    assert next_agent == agents[1]


def test_random_strategy(agents, available_agents):
    strategy = Random()
    current_agent = agents[0]

    tools = strategy.get_tools(current_agent, available_agents)
    assert len(tools) == 1
    assert tools[0].name == "end_turn"

    next_agent = strategy.get_next_agent(current_agent, available_agents)
    assert next_agent in agents


def test_round_robin_strategy(agents, available_agents):
    strategy = RoundRobin()
    current_agent = agents[0]

    tools = strategy.get_tools(current_agent, available_agents)
    assert len(tools) == 1
    assert tools[0].name == "end_turn"

    next_agent = strategy.get_next_agent(current_agent, available_agents)
    assert next_agent == agents[1]

    next_agent = strategy.get_next_agent(agents[2], available_agents)
    assert next_agent == agents[0]


def test_most_busy_strategy(agents, available_agents):
    strategy = MostBusy()
    current_agent = agents[0]

    tools = strategy.get_tools(current_agent, available_agents)
    assert len(tools) == 1
    assert tools[0].name == "end_turn"

    next_agent = strategy.get_next_agent(current_agent, available_agents)
    assert next_agent == agents[0]  # Agent0 has the most tasks (3)


def test_moderated_strategy(agents, available_agents):
    moderator = agents[0]
    strategy = Moderated(moderator=moderator)

    tools = strategy.get_tools(moderator, available_agents)
    assert len(tools) == 1
    assert tools[0].name == "delegate_to_agent"

    tools = strategy.get_tools(agents[1], available_agents)
    assert len(tools) == 1
    assert tools[0].name == "end_turn"

    next_agent = strategy.get_next_agent(moderator, available_agents)
    assert next_agent == moderator

    strategy.next_agent = agents[1]
    next_agent = strategy.get_next_agent(moderator, available_agents)
    assert next_agent == agents[1]

    next_agent = strategy.get_next_agent(agents[1], available_agents)
    assert next_agent == moderator
