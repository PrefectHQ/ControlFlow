from controlflow.agents import Agent, Team


def test_team_agents():
    a1 = Agent(name="a1")
    a2 = Agent(name="a2")
    t = Team(agents=[a1, a2])
    assert t.agents == [a1, a2]


def test_team_with_one_agent():
    a1 = Agent(name="a1")
    t = Team(agents=[a1])
    assert t.agents == [a1]
