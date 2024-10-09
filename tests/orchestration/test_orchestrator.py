from unittest.mock import MagicMock, patch

import pytest

import controlflow.orchestration.conditions
from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.orchestration.orchestrator import Orchestrator
from controlflow.orchestration.turn_strategies import Popcorn, TurnStrategy
from controlflow.tasks.task import Task
from controlflow.utilities.testing import FakeLLM, SimpleTask


class TestOrchestratorLimits:
    @pytest.fixture
    def orchestrator(self, default_fake_llm):
        default_fake_llm.set_responses([dict(name="count_call")])
        self.calls = 0
        self.turns = 0

        class TwoCallTurnStrategy(TurnStrategy):
            """
            A turn strategy that ends a turn after 2 calls
            """

            calls: int = 0

            def get_tools(self, *args, **kwargs):
                return []

            def get_next_agent(self, current_agent, available_agents):
                return current_agent

            def begin_turn(ts_instance):
                self.turns += 1
                super().begin_turn()

            def should_end_turn(ts_self):
                ts_self.calls += 1
                # if this would be the third call, end the turn
                if ts_self.calls >= 3:
                    ts_self.calls = 0
                    return True
                # record a new call for the unit test
                # self.calls += 1
                return False

        def count_call():
            self.calls += 1

        agent = Agent(tools=[count_call])
        task = Task("Test task", agents=[agent])
        flow = Flow()
        orchestrator = Orchestrator(
            tasks=[task],
            flow=flow,
            agent=agent,
            turn_strategy=TwoCallTurnStrategy(),
        )
        return orchestrator

    def test_max_llm_calls(self, orchestrator):
        orchestrator.run(max_llm_calls=5)
        assert self.calls == 5

    def test_max_agent_turns(self, orchestrator):
        orchestrator.run(max_agent_turns=3)
        assert self.calls == 6

    def test_max_llm_calls_and_max_agent_turns(self, orchestrator):
        orchestrator.run(
            max_llm_calls=10,
            max_agent_turns=3,
            model_kwargs={"tool_choice": "required"},
        )
        assert self.calls == 6

    def test_default_limits(self, orchestrator):
        orchestrator.run(model_kwargs={"tool_choice": "required"})
        assert self.calls == 10  # Assuming the default max_llm_calls is 10


class TestOrchestratorCreation:
    def test_create_orchestrator_with_agent(self):
        agent = Agent()
        task = Task("Test task", agents=[agent])
        flow = Flow()
        orchestrator = Orchestrator(tasks=[task], flow=flow, agent=agent)

        assert orchestrator.agent == agent
        assert orchestrator.flow == flow
        assert orchestrator.tasks == [task]

    def test_create_orchestrator_without_agent(self):
        task = Task("Test task")
        flow = Flow()
        orchestrator = Orchestrator(tasks=[task], flow=flow, agent=None)

        assert orchestrator.agent is None
        assert orchestrator.flow == flow
        assert orchestrator.tasks == [task]

    def test_run_sets_agent_if_none(self):
        agent1 = Agent(id="agent1")
        agent2 = Agent(id="agent2")
        task = Task("Test task", agents=[agent1, agent2])
        flow = Flow()
        turn_strategy = Popcorn()
        orchestrator = Orchestrator(
            tasks=[task], flow=flow, agent=None, turn_strategy=turn_strategy
        )

        assert orchestrator.agent is None

        orchestrator.run(max_agent_turns=0)

        assert orchestrator.agent is not None
        assert orchestrator.agent in [agent1, agent2]

    def test_run_keeps_existing_agent_if_set(self):
        agent1 = Agent(id="agent1")
        agent2 = Agent(id="agent2")
        task = Task("Test task", agents=[agent1, agent2])
        flow = Flow()
        turn_strategy = Popcorn()
        orchestrator = Orchestrator(
            tasks=[task], flow=flow, agent=agent1, turn_strategy=turn_strategy
        )

        assert orchestrator.agent == agent1

        orchestrator.run(max_agent_turns=0)

        assert orchestrator.agent == agent1


class TestRunEndConditions:
    def test_run_until_all_complete(self, monkeypatch):
        task1 = SimpleTask()
        task2 = SimpleTask()
        orchestrator = Orchestrator(tasks=[task1, task2], flow=Flow(), agent=Agent())

        # Mock the run_agent_turn method
        def mock_run_agent_turn(*args, **kwargs):
            task1.mark_successful()
            task2.mark_successful()
            return 1

        monkeypatch.setitem(
            orchestrator.__dict__,
            "run_agent_turn",
            MagicMock(side_effect=mock_run_agent_turn),
        )

        orchestrator.run(run_until=controlflow.orchestration.conditions.AllComplete())

        assert all(task.is_complete() for task in orchestrator.tasks)

    def test_run_until_any_complete(self, monkeypatch):
        task1 = SimpleTask()
        task2 = SimpleTask()
        orchestrator = Orchestrator(tasks=[task1, task2], flow=Flow(), agent=Agent())

        # Mock the run_agent_turn method
        def mock_run_agent_turn(*args, **kwargs):
            task1.mark_successful()
            return 1

        monkeypatch.setitem(
            orchestrator.__dict__,
            "run_agent_turn",
            MagicMock(side_effect=mock_run_agent_turn),
        )

        orchestrator.run(run_until=controlflow.orchestration.conditions.AnyComplete())

        assert any(task.is_complete() for task in orchestrator.tasks)

    def test_run_until_fn_condition(self, monkeypatch):
        task1 = SimpleTask()
        task2 = SimpleTask()
        orchestrator = Orchestrator(tasks=[task1, task2], flow=Flow(), agent=Agent())

        # Mock the run_agent_turn method
        def mock_run_agent_turn(*args, **kwargs):
            task2.mark_successful()
            return 1

        monkeypatch.setitem(
            orchestrator.__dict__,
            "run_agent_turn",
            MagicMock(side_effect=mock_run_agent_turn),
        )

        orchestrator.run(
            run_until=controlflow.orchestration.conditions.FnCondition(
                lambda context: context.orchestrator.tasks[1].is_complete()
            )
        )

        assert task2.is_complete()

    def test_run_until_lambda_condition(self, monkeypatch):
        task1 = SimpleTask()
        task2 = SimpleTask()
        orchestrator = Orchestrator(tasks=[task1, task2], flow=Flow(), agent=Agent())

        # Mock the run_agent_turn method
        def mock_run_agent_turn(*args, **kwargs):
            task2.mark_successful()
            return 1

        monkeypatch.setitem(
            orchestrator.__dict__,
            "run_agent_turn",
            MagicMock(side_effect=mock_run_agent_turn),
        )

        orchestrator.run(
            run_until=lambda context: context.orchestrator.tasks[1].is_complete()
        )

        assert task2.is_complete()

    def test_compound_condition(self, monkeypatch):
        task1 = SimpleTask()
        task2 = SimpleTask()
        orchestrator = Orchestrator(tasks=[task1, task2], flow=Flow(), agent=Agent())

        # Mock the run_agent_turn method
        def mock_run_agent_turn(*args, **kwargs):
            return 1

        monkeypatch.setitem(
            orchestrator.__dict__,
            "run_agent_turn",
            MagicMock(side_effect=mock_run_agent_turn),
        )

        orchestrator.run(
            run_until=(
                # this condition will always fail
                controlflow.orchestration.conditions.FnCondition(lambda context: False)
                |
                # this condition will always pass
                controlflow.orchestration.conditions.FnCondition(lambda context: True)
            )
        )

        # assert to prove we reach this point and the run stopped
        assert True
