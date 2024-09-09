import pytest

from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.orchestration.orchestrator import Orchestrator
from controlflow.orchestration.turn_strategies import (  # Add this import
    Popcorn,
    TurnStrategy,
)
from controlflow.tasks.task import Task


class TestOrchestratorLimits:
    call_count = 0
    turn_count = 0

    @pytest.fixture
    def mocked_orchestrator(self, default_fake_llm):
        # Reset counts at the start of each test
        self.call_count = 0
        self.turn_count = 0

        class TwoCallTurnStrategy(TurnStrategy):
            calls: int = 0

            def get_tools(self, *args, **kwargs):
                return []

            def get_next_agent(self, current_agent, available_agents):
                return current_agent

            def begin_turn(ts_instance):
                self.turn_count += 1
                super().begin_turn()

            def should_end_turn(ts_instance):
                ts_instance.calls += 1
                # if this would be the third call, end the turn
                if ts_instance.calls >= 3:
                    ts_instance.calls = 0
                    return True
                # record a new call for the unit test
                self.call_count += 1
                return False

        agent = Agent()
        task = Task("Test task", agents=[agent])
        flow = Flow()
        orchestrator = Orchestrator(
            tasks=[task], flow=flow, agent=agent, turn_strategy=TwoCallTurnStrategy()
        )

        return orchestrator

    def test_default_limits(self, mocked_orchestrator):
        mocked_orchestrator.run()

        assert self.turn_count == 5
        assert self.call_count == 10

    @pytest.mark.parametrize(
        "max_agent_turns, max_llm_calls, expected_turns, expected_calls",
        [
            (1, 1, 1, 1),
            (1, 2, 1, 2),
            (5, 3, 2, 3),
            (3, 12, 3, 6),
        ],
    )
    def test_custom_limits(
        self,
        mocked_orchestrator,
        max_agent_turns,
        max_llm_calls,
        expected_turns,
        expected_calls,
    ):
        mocked_orchestrator.run(
            max_agent_turns=max_agent_turns, max_llm_calls=max_llm_calls
        )

        assert self.turn_count == expected_turns
        assert self.call_count == expected_calls

    def test_task_limit(self, mocked_orchestrator):
        task = Task("Test task", max_llm_calls=5, agents=[mocked_orchestrator.agent])
        mocked_orchestrator.tasks = [task]
        mocked_orchestrator.run()
        assert task.is_failed()
        assert self.turn_count == 3
        # Note: the call count will be 6 because the orchestrator call count is
        # incremented in "should_end_turn" which is called before the task's
        # call count is evaluated
        assert self.call_count == 6

    def test_task_lifetime_limit(self, mocked_orchestrator):
        task = Task("Test task", max_llm_calls=5, agents=[mocked_orchestrator.agent])
        mocked_orchestrator.tasks = [task]
        mocked_orchestrator.run(max_agent_turns=1)
        assert task.is_incomplete()
        mocked_orchestrator.run(max_agent_turns=1)
        assert task.is_incomplete()
        mocked_orchestrator.run(max_agent_turns=1)
        assert task.is_failed()

        assert self.turn_count == 3
        # Note: the call count will be 6 because the orchestrator call count is
        # incremented in "should_end_turn" which is called before the task's
        # call count is evaluated
        assert self.call_count == 6


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
