import pytest

import controlflow
from controlflow.agents.agent import Agent
from controlflow.run import run, run_async, run_tasks, run_tasks_async


def test_run():
    result = run("what's 2 + 2", result_type=int)
    assert result == 4


async def test_run_async():
    result = await run_async("what's 2 + 2", result_type=int)
    assert result == 4


class TestLimits:
    call_count = 0

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch, default_fake_llm):
        self.call_count = 0

        original_run_model = Agent._run_model
        original_run_model_async = Agent._run_model_async

        def mock_run_model(*args, **kwargs):
            self.call_count += 1
            return original_run_model(*args, **kwargs)

        async def mock_run_model_async(*args, **kwargs):
            self.call_count += 1
            async for event in original_run_model_async(*args, **kwargs):
                yield event

        monkeypatch.setattr(Agent, "_run_model", mock_run_model)
        monkeypatch.setattr(Agent, "_run_model_async", mock_run_model_async)

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
        self,
        max_turns,
        max_calls_per_turn,
        expected_calls,
    ):
        run(
            "send messages",
            max_calls_per_turn=max_calls_per_turn,
            max_turns=max_turns,
        )

        assert self.call_count == expected_calls

    @pytest.mark.parametrize(
        "max_turns, max_calls_per_turn, expected_calls",
        [
            (1, 1, 1),
            (1, 2, 2),
            (2, 1, 2),
            (3, 2, 6),
        ],
    )
    async def test_run_async_with_limits(
        self,
        max_turns,
        max_calls_per_turn,
        expected_calls,
    ):
        await run_async(
            "send messages",
            max_calls_per_turn=max_calls_per_turn,
            max_turns=max_turns,
        )

        assert self.call_count == expected_calls

    @pytest.mark.parametrize(
        "max_turns, max_calls_per_turn, expected_calls",
        [
            (1, 1, 1),
            (1, 2, 2),
            (2, 1, 2),
            (3, 2, 6),
        ],
    )
    def test_run_task_with_limits(
        self,
        max_turns,
        max_calls_per_turn,
        expected_calls,
    ):
        run_tasks(
            tasks=[
                controlflow.Task("send messages"),
                controlflow.Task("send messages"),
            ],
            max_calls_per_turn=max_calls_per_turn,
            max_turns=max_turns,
        )

        assert self.call_count == expected_calls

    @pytest.mark.parametrize(
        "max_turns, max_calls_per_turn, expected_calls",
        [
            (1, 1, 1),
            (1, 2, 2),
            (2, 1, 2),
            (3, 2, 6),
        ],
    )
    async def test_run_task_async_with_limits(
        self,
        max_turns,
        max_calls_per_turn,
        expected_calls,
    ):
        await run_tasks_async(
            tasks=[
                controlflow.Task("send messages"),
                controlflow.Task("send messages"),
            ],
            max_calls_per_turn=max_calls_per_turn,
            max_turns=max_turns,
        )

        assert self.call_count == expected_calls
