import controlflow
import pytest
from controlflow import Task
from controlflow.decorators import flow, task
from controlflow.settings import temporary_settings


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


class TestTaskDecorator:
    pass


@pytest.mark.usefixtures("mock_controller")
class TestTaskEagerMode:
    def test_eager_mode_enabled_by_default(self):
        assert controlflow.settings.eager_mode is True

    def test_task_eager_mode(self, mock_controller_run_agent):
        @task
        def return_42() -> int:
            """Return the number 42"""
            pass

        return_42()
        assert mock_controller_run_agent.call_count == 1

    def test_task_eager_mode_off(self, mock_controller_run_agent):
        @task(eager=False)
        def return_42() -> int:
            """Return the number 42"""
            pass

        result = return_42()
        assert mock_controller_run_agent.call_count == 0
        assert isinstance(result, Task)
        assert result.objective == "return_42"
        assert result.result_type == int
        assert result.instructions == "Return the number 42"

    def test_task_eager_mode_loads_default(self, mock_controller_run_agent):
        with temporary_settings(eager_mode=False):

            @task
            def return_42() -> int:
                """Return the number 42"""
                pass

        result = return_42()
        assert mock_controller_run_agent.call_count == 0
        assert isinstance(result, Task)
        assert result.objective == "return_42"
        assert result.result_type == int
        assert result.instructions == "Return the number 42"

    @pytest.mark.parametrize("eager_mode", [True, False])
    def test_override_eager_mode_at_call_time(
        self, mock_controller_run_agent, eager_mode
    ):
        with temporary_settings(eager_mode=eager_mode):

            @task
            def return_42() -> int:
                """Return the number 42"""
                pass

        return_42(eager_=not eager_mode)
        if eager_mode:
            assert mock_controller_run_agent.call_count == 0
        else:
            assert mock_controller_run_agent.call_count == 1


@pytest.mark.usefixtures("mock_controller")
class TestFlowEagerMode:
    def test_flow_eager_mode(self, mock_controller_run_agent):
        @flow
        def test_flow():
            task = Task("say hello", result="hello")
            return task

        result = test_flow()
        assert mock_controller_run_agent.call_count == 1
        assert result == "hello"

    def test_flow_eager_mode_off(self, mock_controller_run_agent):
        @flow(eager=False)
        def test_flow():
            task = Task("say hello", result="hello")
            return task

        result = test_flow()
        assert mock_controller_run_agent.call_count == 0
        assert isinstance(result, Task)
        assert result.objective == "say hello"
        assert result.result == "hello"

    def test_flow_eager_mode_off_doesnt_affect_tasks_with_eager_mode_on(
        self, mock_controller_run_agent
    ):
        @task(eager=True)
        def return_42() -> int:
            """Return the number 42"""
            pass

        @flow(eager=False)
        def test_flow():
            result = return_42()
            return result

        result = test_flow()
        assert mock_controller_run_agent.call_count == 1
        assert not isinstance(result, Task)
