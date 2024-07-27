from functools import partial

import controlflow
from controlflow.agents import Agent, Team
from controlflow.utilities.testing import SimpleTask


class TestTeam:
    def test_team_agents(self):
        a1 = Agent(name="a1")
        a2 = Agent(name="a2")
        t = Team(agents=[a1, a2])
        assert t.agents == [a1, a2]

    def test_team_with_one_agent(self):
        a1 = Agent(name="a1")
        t = Team(agents=[a1])
        assert t.agents == [a1]

    def test_stable_id(self):
        assert Team(agents=[Agent(name="a1")], name="Test Team").id == "01874907"
        assert Team(agents=[Agent(name="a2")], name="Test Team").id == "8e62ddc8"


class TestDefaultTeam:
    def test_default_team(self):
        a1 = Agent(name="a1")
        a2 = Agent(name="a2")
        team_callable = controlflow.defaults.team
        team = team_callable(agents=[a1, a2])
        assert team.name == "Agents"
        assert team.agents == [a1, a2]

    def test_default_team_can_be_assigned(self):
        a1 = Agent(name="a1")
        a2 = Agent(name="a2")
        controlflow.defaults.team = partial(Team, name="New Team")

        task = SimpleTask(agents=[a1, a2])
        assert isinstance(task.agent, Team)
        assert task.agent.agents == [a1, a2]
        assert task.agent.name == "New Team"
