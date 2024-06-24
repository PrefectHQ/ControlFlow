from unittest.mock import AsyncMock

import pytest
from controlflow.core.agent import Agent, get_default_agent
from controlflow.core.flow import Flow
from controlflow.core.graph import EdgeType
from controlflow.core.task import Task, TaskStatus
from controlflow.settings import temporary_settings
from controlflow.utilities.context import ctx


def test_context_open_and_close():
    assert ctx.get("tasks") == []
    with Task("a") as ta:
        assert ctx.get("tasks") == [ta]
        with Task("b") as tb:
            assert ctx.get("tasks") == [ta, tb]
        assert ctx.get("tasks") == [ta]
    assert ctx.get("tasks") == []


def test_task_initialization():
    task = Task(objective="Test objective")
    assert task.objective == "Test objective"
    assert task.status == TaskStatus.INCOMPLETE
    assert task.result is None
    assert task.error is None


def test_task_dependencies():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    assert task1 in task2.depends_on
    assert task2 in task1._downstreams


def test_task_subtasks():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", parent=task1)
    assert task2 in task1._subtasks
    assert task2.parent is task1


def test_task_parent_context():
    with Task("grandparent") as task1:
        with Task("parent") as task2:
            task3 = Task("child")

    assert task3.parent is task2
    assert task2.parent is task1
    assert task1.parent is None

    assert task1._subtasks == {task2}
    assert task2._subtasks == {task3}
    assert task3._subtasks == set()


def test_task_agent_assignment():
    agent = Agent(name="Test Agent")
    task = Task(objective="Test objective", agents=[agent])
    assert agent in task.agents


def test_task_bad_agent_assignment():
    with pytest.raises(ValueError):
        Task(objective="Test objective", agents=[])


def test_task_loads_agent_from_parent():
    agent = Agent(name="Test Agent")
    with Task("parent", agents=[agent]):
        child = Task("child")

    assert child.agents is None
    assert child.get_agents() == [agent]


def test_task_loads_agent_from_flow():
    def_agent = get_default_agent()
    agent = Agent(name="Test Agent")
    with Flow(agents=[agent]):
        task = Task("task")

        assert task.agents is None
        assert task.get_agents() == [agent]

    # outside the flow context, pick up the default agent
    assert task.get_agents() == [def_agent]


def test_task_loads_agent_from_default_if_none_otherwise():
    agent = get_default_agent()
    task = Task("task")

    assert task.agents is None
    assert task.get_agents() == [agent]


def test_task_loads_agent_from_parent_before_flow():
    agent1 = Agent(name="Test Agent 1")
    agent2 = Agent(name="Test Agent 2")
    with Flow(agents=[agent1]):
        with Task("parent", agents=[agent2]):
            child = Task("child")

    assert child.agents is None
    assert child.get_agents() == [agent2]


def test_task_tracking(mock_controller_run_agent):
    with Flow() as flow:
        task = Task(objective="Test objective")
        assert task in flow._tasks.values()


def test_task_tracking_on_call(mock_controller_run_agent):
    task = Task(objective="Test objective")
    with Flow() as flow:
        task.run_once()
    assert task in flow._tasks.values()


def test_task_status_transitions():
    task = Task(objective="Test objective")
    assert task.is_incomplete()
    assert not task.is_complete()
    assert not task.is_successful()
    assert not task.is_failed()
    assert not task.is_skipped()

    task.mark_successful()
    assert not task.is_incomplete()
    assert task.is_complete()
    assert task.is_successful()
    assert not task.is_failed()
    assert not task.is_skipped()

    task = Task(objective="Test objective")
    task.mark_failed()
    assert not task.is_incomplete()
    assert task.is_complete()
    assert not task.is_successful()
    assert task.is_failed()
    assert not task.is_skipped()

    task = Task(objective="Test objective")
    task.mark_skipped()
    assert not task.is_incomplete()
    assert task.is_complete()
    assert not task.is_successful()
    assert not task.is_failed()
    assert task.is_skipped()


def test_validate_upstream_dependencies_on_success():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    with pytest.raises(ValueError, match="cannot be marked successful"):
        task2.mark_successful()
    task1.mark_successful()
    task2.mark_successful()


def test_validate_subtask_dependencies_on_success():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", parent=task1)
    with pytest.raises(ValueError, match="cannot be marked successful"):
        task1.mark_successful()
    task2.mark_successful()
    task1.mark_successful()


def test_task_ready():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    assert not task2.is_ready

    task1.mark_successful()
    assert task2.is_ready


def test_task_hash():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2")
    assert hash(task1) != hash(task2)


def test_task_tools():
    task = Task(objective="Test objective")
    tools = task.get_tools()
    assert any(tool.function.name == f"mark_task_{task.id}_failed" for tool in tools)
    assert any(
        tool.function.name == f"mark_task_{task.id}_successful" for tool in tools
    )

    task.mark_successful()
    tools = task.get_tools()
    assert not any(
        tool.function.name == f"mark_task_{task.id}_failed" for tool in tools
    )
    assert not any(
        tool.function.name == f"mark_task_{task.id}_successful" for tool in tools
    )


class TestTaskToGraph:
    def test_single_task_graph(self):
        task = Task(objective="Test objective")
        graph = task.as_graph()
        assert len(graph.tasks) == 1
        assert task in graph.tasks
        assert len(graph.edges) == 0

    def test_task_with_subtasks_graph(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", parent=task1)
        graph = task1.as_graph()
        assert len(graph.tasks) == 2
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert len(graph.edges) == 1
        assert any(
            edge.upstream == task2
            and edge.downstream == task1
            and edge.type == EdgeType.SUBTASK
            for edge in graph.edges
        )

    def test_task_with_dependencies_graph(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        graph = task2.as_graph()
        assert len(graph.tasks) == 2
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert len(graph.edges) == 1
        assert any(
            edge.upstream == task1
            and edge.downstream == task2
            and edge.type == EdgeType.DEPENDENCY
            for edge in graph.edges
        )

    def test_task_with_subtasks_and_dependencies_graph(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = task2.as_graph()
        assert len(graph.tasks) == 3
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert task3 in graph.tasks
        assert len(graph.edges) == 2
        assert any(
            edge.upstream == task1
            and edge.downstream == task2
            and edge.type == EdgeType.DEPENDENCY
            for edge in graph.edges
        )
        assert any(
            edge.upstream == task3
            and edge.downstream == task2
            and edge.type == EdgeType.SUBTASK
            for edge in graph.edges
        )


@pytest.mark.usefixtures("mock_run")
class TestTaskRun:
    def test_run_task_max_iterations(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        with Flow():
            with pytest.raises(ValueError):
                task.run()

        assert mock_run.await_count == 3

    def test_run_task_mark_successful(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        def mark_complete():
            task.mark_successful()

        mock_run.side_effect = mark_complete
        with Flow():
            result = task.run()
        assert task.is_successful()
        assert result is None

    def test_run_task_mark_successful_with_result(self, mock_run: AsyncMock):
        task = Task[int](objective="Say hello")

        def mark_complete():
            task.mark_successful(result=42)

        mock_run.side_effect = mark_complete
        with Flow():
            result = task.run()
        assert task.is_successful()
        assert result == 42

    def test_run_task_mark_failed(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        def mark_complete():
            task.mark_failed(message="Failed to say hello")

        mock_run.side_effect = mark_complete
        with Flow():
            with pytest.raises(ValueError):
                task.run()
        assert task.is_failed()
        assert task.error == "Failed to say hello"

    def test_run_task_outside_flow(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        def mark_complete():
            task.mark_successful()

        mock_run.side_effect = mark_complete
        result = task.run()
        assert task.is_successful()
        assert result is None

    def test_run_task_outside_flow_fails_if_strict_flows_enforced(
        self, mock_run: AsyncMock
    ):
        task = Task(objective="Say hello")

        with temporary_settings(strict_flow_context=True):
            with pytest.raises(ValueError):
                task.run()

    def test_task_run_once_outside_flow_fails(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        with pytest.raises(ValueError):
            task.run_once()

    def test_task_run_once_with_passed_flow(self, mock_run: AsyncMock):
        task = Task(objective="Say hello")

        def mark_complete():
            task.mark_successful()

        mock_run.side_effect = mark_complete
        flow = Flow()
        while task.is_incomplete():
            task.run_once(flow=flow)
        assert task.is_successful()
        assert task.result is None
