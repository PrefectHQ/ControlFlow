import pytest

import controlflow
from controlflow.agents import Agent
from controlflow.flows import Flow
from controlflow.orchestration.orchestrator import Orchestrator
from controlflow.tasks.task import Task


@pytest.fixture
def mocked_orchestrator(monkeypatch):
    agent = Agent()
    task = Task("Test task", agents=[agent])
    flow = Flow()
    orchestrator = Orchestrator(tasks=[task], flow=flow, agent=agent)

    call_count = 0
    turn_count = 0
    original_run_model = Agent._run_model
    original_run_turn = Orchestrator._run_turn

    def mock_run_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_run_model(*args, **kwargs)

    def mock_run_turn(*args, **kwargs):
        nonlocal turn_count
        turn_count += 1
        return original_run_turn(*args, **kwargs)

    monkeypatch.setattr(Agent, "_run_model", mock_run_model)
    monkeypatch.setattr(Orchestrator, "_run_turn", mock_run_turn)

    return orchestrator, lambda: call_count, lambda: turn_count


class TestOrchestratorLimits:
    def test_default_limits(self, mocked_orchestrator, default_fake_llm, monkeypatch):
        monkeypatch.setattr(controlflow.defaults, "model", default_fake_llm)
        orchestrator, get_call_count, get_turn_count = mocked_orchestrator

        orchestrator.run()

        assert get_turn_count() == controlflow.settings.orchestrator_max_turns
        assert (
            get_call_count()
            == controlflow.settings.orchestrator_max_turns
            * controlflow.settings.orchestrator_max_calls_per_turn
        )

    @pytest.mark.parametrize(
        "max_turns, max_calls_per_turn, expected_calls",
        [
            (1, 1, 1),
            (1, 2, 2),
            (2, 1, 2),
            (3, 2, 6),
        ],
    )
    def test_custom_limits(
        self,
        mocked_orchestrator,
        default_fake_llm,
        monkeypatch,
        max_turns,
        max_calls_per_turn,
        expected_calls,
    ):
        monkeypatch.setattr(controlflow.defaults, "model", default_fake_llm)
        orchestrator, get_call_count, _ = mocked_orchestrator

        orchestrator.run(max_turns=max_turns, max_calls_per_turn=max_calls_per_turn)

        assert get_call_count() == expected_calls

    def test_max_turns_reached(
        self, mocked_orchestrator, default_fake_llm, monkeypatch
    ):
        monkeypatch.setattr(controlflow.defaults, "model", default_fake_llm)
        orchestrator, _, get_turn_count = mocked_orchestrator

        orchestrator.run(max_turns=5)

        assert get_turn_count() == 5

    def test_max_calls_per_turn_reached(
        self, mocked_orchestrator, default_fake_llm, monkeypatch
    ):
        monkeypatch.setattr(controlflow.defaults, "model", default_fake_llm)
        orchestrator, get_call_count, _ = mocked_orchestrator

        orchestrator.run(max_calls_per_turn=3)

        assert get_call_count() == 3 * controlflow.settings.orchestrator_max_turns


def test_run():
    result = controlflow.run("what's 2 + 2", result_type=int)
    assert result == 4


async def test_run_async():
    result = await controlflow.run_async("what's 2 + 2", result_type=int)
    assert result == 4


@pytest.mark.parametrize(
    "max_turns, max_calls_per_turn, expected_calls",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (3, 2, 6),
    ],
)
def test_run_with_limits(
    monkeypatch, default_fake_llm, max_turns, max_calls_per_turn, expected_calls
):
    call_count = 0
    original_run_model = Agent._run_model

    def mock_run_model(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_run_model(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "_run_model", mock_run_model)

    controlflow.run(
        "send messages",
        max_calls_per_turn=max_calls_per_turn,
        max_turns=max_turns,
    )

    assert call_count == expected_calls
