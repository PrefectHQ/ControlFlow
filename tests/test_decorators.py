import pytest
from controlflow import Task
from controlflow.decorators import flow


@pytest.mark.usefixtures("mock_controller")
class TestFlowDecorator:
    def test_flow_decorator(self):
        @flow
        def test_flow():
            return 1

        result = test_flow()
        assert result == 1

    def test_flow_decorator_runs_all_tasks(self):
        tasks: list[Task] = []

        @flow
        def test_flow():
            task = Task(
                "say hello",
                result_type=str,
                result="Task completed successfully",
            )
            tasks.append(task)

        result = test_flow()
        assert result is None
        assert tasks[0].is_successful()
        assert tasks[0].result == "Task completed successfully"

    def test_flow_decorator_resolves_all_tasks(self):
        @flow
        def test_flow():
            task1 = Task("say hello", result="hello")
            task2 = Task("say goodbye", result="goodbye")
            task3 = Task("say goodnight", result="goodnight")
            return dict(a=task1, b=[task2], c=dict(x=dict(y=[[task3]])))

        result = test_flow()
        assert result == dict(
            a="hello", b=["goodbye"], c=dict(x=dict(y=[["goodnight"]]))
        )

    def test_manually_run_task_in_flow(self):
        @flow
        def test_flow():
            task = Task("say hello", result="hello")
            task.run()
            return task.result

        result = test_flow()
        assert result == "hello"
