from control_flow.core.agent import Agent
from pytest import patch


class TestAgent:
    pass


class TestAgentRun:
    def test_agent_run(self):
        with patch(
            "control_flow.core.controller.Controller._get_prefect_run_agent_task"
        ) as mock_task:
            agent = Agent()
            agent.run()
            mock_task.assert_called_once()
