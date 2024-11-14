import pytest

import controlflow
from controlflow import Stream, instructions
from controlflow.events.base import Event
from controlflow.events.events import AgentMessage
from controlflow.llm.messages import AIMessage
from controlflow.orchestration.conditions import AnyComplete, AnyFailed, MaxLLMCalls
from controlflow.orchestration.handler import Handler
from controlflow.run import run, run_async, run_tasks, run_tasks_async
from controlflow.tasks.task import Task
from tests.fixtures.controlflow import default_fake_llm


class TestHandlers:
    class ExampleHandler(Handler):
        def __init__(self):
            self.events = []
            self.agent_messages = []

        def on_event(self, event: Event):
            self.events.append(event)

        def on_agent_message(self, event: AgentMessage):
            self.agent_messages.append(event)

    def test_run_with_handlers(self, default_fake_llm):
        handler = self.ExampleHandler()
        run("what's 2 + 2", result_type=int, handlers=[handler], max_llm_calls=1)
        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1

    async def test_run_async_with_handlers(self, default_fake_llm):
        handler = self.ExampleHandler()
        await run_async(
            "what's 2 + 2", result_type=int, handlers=[handler], max_llm_calls=1
        )

        assert len(handler.events) > 0
        assert len(handler.agent_messages) == 1


def test_run():
    result = run("what's 2 + 2", result_type=int)
    assert result == 4


async def test_run_async():
    result = await run_async("what's 2 + 2", result_type=int)
    assert result == 4


class TestRunUntil:
    def test_any_complete(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")

        with instructions("complete only task 2"):
            run_tasks([task1, task2], run_until=AnyComplete())

        assert task2.is_complete()
        assert task1.is_incomplete()

    def test_any_failed(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")

        with instructions("fail only task 2"):
            run_tasks([task1, task2], run_until=AnyFailed(), raise_on_failure=False)

        assert task2.is_failed()
        assert not task1.is_failed()

    def test_max_llm_calls(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")

        with instructions("say hi but do not complete any tasks"):
            run_tasks([task1, task2], run_until=MaxLLMCalls(1))

        assert task2.is_incomplete()
        assert task1.is_incomplete()

    def test_min_complete(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")
        task3 = Task("Task 3")

        with instructions("complete tasks 1 and 2"):
            run_tasks([task1, task2, task3], run_until=AnyComplete(min_complete=2))

        assert task1.is_complete()
        assert task2.is_complete()
        assert task3.is_incomplete()

    def test_min_failed(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")
        task3 = Task("Task 3")

        with instructions("fail tasks 1 and 3. Don't work on task 2."):
            run_tasks(
                [task1, task2, task3],
                run_until=AnyFailed(min_failed=2),
                raise_on_failure=False,
            )

        assert task1.is_failed()
        assert task2.is_incomplete()
        assert task3.is_failed()


class TestRunUntilAsync:
    async def test_any_complete(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")

        with instructions("complete only task 2"):
            await run_tasks_async([task1, task2], run_until=AnyComplete())

        assert task2.is_complete()
        assert task1.is_incomplete()

    async def test_any_failed(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")

        with instructions("fail only task 2"):
            await run_tasks_async(
                [task1, task2], run_until=AnyFailed(), raise_on_failure=False
            )

        assert task2.is_failed()
        assert not task1.is_failed()

    async def test_max_llm_calls(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")

        with instructions("say hi but do not complete any tasks"):
            await run_tasks_async([task1, task2], run_until=MaxLLMCalls(1))

        assert task2.is_incomplete()
        assert task1.is_incomplete()

    async def test_min_complete(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")
        task3 = Task("Task 3")

        with instructions("complete tasks 1 and 2"):
            await run_tasks_async(
                [task1, task2, task3], run_until=AnyComplete(min_complete=2)
            )

        assert task1.is_complete()
        assert task2.is_complete()
        assert task3.is_incomplete()

    async def test_min_failed(self):
        task1 = Task("Task 1")
        task2 = Task("Task 2")
        task3 = Task("Task 3")

        with instructions("fail tasks 1 and 3. Don't work on task 2."):
            await run_tasks_async(
                [task1, task2, task3],
                run_until=AnyFailed(min_failed=2),
                raise_on_failure=False,
            )

        assert task1.is_failed()
        assert task2.is_incomplete()
        assert task3.is_failed()


class TestRunStreaming:
    @pytest.fixture
    def task(self, default_fake_llm):
        task = controlflow.Task("say hello", id="12345")

        response = AIMessage(
            id="run-2af8bb73-661f-4ec3-92ff-d7d8e3074926",
            name="Marvin",
            role="ai",
            content="",
            tool_calls=[
                {
                    "name": "mark_task_12345_successful",
                    "args": {"task_result": "Hello!"},
                    "id": "call_ZEPdV8mCgeBe5UHjKzm6e3pe",
                    "type": "tool_call",
                }
            ],
        )

        default_fake_llm.set_responses(["Hello!", response])

        return task

    def test_stream_all(self, default_fake_llm):
        result = run("what's 2 + 2", stream=True, max_llm_calls=1)
        r = list(result)
        assert len(r) > 5

    def test_stream_task(self, task):
        result = list(task.run(stream=True))
        assert result[0][0].event == "orchestrator-start"
        assert result[1][0].event == "agent-turn-start"
        assert result[-1][0].event == "orchestrator-end"
        assert any(r[0].event == "agent-message" for r in result)
        assert any(r[0].event == "agent-message-delta" for r in result)
        assert any(r[0].event == "agent-content" for r in result)
        assert any(r[0].event == "agent-content-delta" for r in result)
        assert any(r[0].event == "agent-tool-call" for r in result)

    def test_stream_content(self, task):
        result = list(task.run(stream=Stream.CONTENT))
        assert all(
            r[0].event in ("agent-content", "agent-content-delta") for r in result
        )

    def test_stream_tools(self, task):
        result = list(task.run(stream=Stream.TOOLS))
        assert all(
            r[0].event in ("agent-tool-call", "agent-tool-call-delta", "tool-result")
            for r in result
        )

    def test_stream_results(self, task):
        result = list(task.run(stream=Stream.COMPLETION_TOOLS))
        assert all(
            r[0].event in ("agent-tool-call", "agent-tool-call-delta", "tool-result")
            for r in result
        )
