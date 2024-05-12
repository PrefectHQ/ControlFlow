from unittest.mock import patch

from controlflow.core.agent import Agent
from controlflow.core.task import Task


class TestAgent:
    pass


class TestAgentRun:
    def test_agent_run(self):
        with patch(
            "controlflow.core.controller.Controller._get_prefect_run_agent_task"
        ) as mock_task:
            agent = Agent()
            agent.run()
            mock_task.assert_called_once()

    def test_agent_run_with_task(self):
        task = Task("say hello")
        agent = Agent()
        agent.run(tasks=[task])
