import pytest

import controlflow
from controlflow.agents.agent import Agent


def test_run():
    result = controlflow.run("what's 2 + 2", result_type=int)
    assert result == 4


async def test_run_async():
    result = await controlflow.run_async("what's 2 + 2", result_type=int)
    assert result == 4


@pytest.mark.parametrize(
    "turns, calls_per_turn, expected_calls",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (3, 2, 6),
    ],
)
def test_run_with_limits(
    monkeypatch, default_fake_llm, turns, calls_per_turn, expected_calls
):
    call_count = 0
    original_run_model = Agent._run_model

    def mock_run_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_run_model(*args, **kwargs)

    monkeypatch.setattr(Agent, "_run_model", mock_run_model)

    controlflow.run(
        "send messages",
        max_calls_per_turn=calls_per_turn,
        max_turns=turns,
    )

    assert call_count == expected_calls
