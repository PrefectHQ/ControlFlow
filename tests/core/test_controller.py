from unittest.mock import AsyncMock

import pytest
from control_flow.core.agent import Agent
from control_flow.core.controller.controller import Controller
from control_flow.core.flow import Flow
from control_flow.core.graph import EdgeType
from control_flow.core.task import Task


class TestController:
    @pytest.fixture
    def flow(self):
        return Flow()

    @pytest.fixture
    def agent(self):
        return Agent(name="Test Agent")

    @pytest.fixture
    def task(self):
        return Task(objective="Test Task")

    def test_controller_initialization(self, flow, agent, task):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        assert controller.flow == flow
        assert controller.tasks == [task]
        assert controller.agents == [agent]
        assert controller.run_dependencies is True
        assert len(controller.context) == 0
        assert len(controller.graph.tasks) == 1
        assert len(controller.graph.edges) == 0

    def test_controller_missing_tasks(self, flow):
        with pytest.raises(ValueError, match="At least one task is required."):
            Controller(flow=flow, tasks=[])

    async def test_run_agent(self, flow, agent, task, monkeypatch):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        mocked_run = AsyncMock()
        monkeypatch.setattr(Agent, "run", mocked_run)
        await controller._run_agent(agent, tasks=[task])
        mocked_run.assert_called_once_with(tasks=[task])

    async def test_run_once(self, flow, agent, task, monkeypatch):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        mocked_run_agent = AsyncMock()
        monkeypatch.setattr(Controller, "_run_agent", mocked_run_agent)
        await controller.run_once_async()
        mocked_run_agent.assert_called_once_with(agent, tasks=[task])

    def test_create_end_run_tool(self, flow, agent, task):
        controller = Controller(flow=flow, tasks=[task], agents=[agent])
        end_run_tool = controller._create_end_run_tool()
        assert end_run_tool.function.name == "end_run"
        assert end_run_tool.function.description.startswith("End your turn")

    def test_controller_graph_creation(self, flow, agent):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        controller = Controller(flow=flow, tasks=[task1, task2], agents=[agent])
        assert len(controller.graph.tasks) == 2
        assert len(controller.graph.edges) == 1
        assert controller.graph.edges.pop().type == EdgeType.dependency

    def test_controller_agent_selection(self, flow, monkeypatch):
        agent1 = Agent(name="Agent 1")
        agent2 = Agent(name="Agent 2")
        task = Task(objective="Test Task", agents=[agent1, agent2])
        controller = Controller(flow=flow, tasks=[task], agents=[agent1, agent2])
        mocked_marvin_moderator = AsyncMock(return_value=agent1)
        monkeypatch.setattr(
            "control_flow.core.controller.moderators.marvin_moderator",
            mocked_marvin_moderator,
        )
        assert controller.agents == [agent1, agent2]

    async def test_controller_run_dependencies(self, flow, agent):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        controller = Controller(flow=flow, tasks=[task2], agents=[agent])
        mocked_run_agent = AsyncMock()
        controller._run_agent = mocked_run_agent
        await controller.run_once_async()
        mocked_run_agent.assert_called_once_with(agent, tasks=[task1, task2])
