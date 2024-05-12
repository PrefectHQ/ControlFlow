from control_flow.core.agent import Agent
from control_flow.core.flow import Flow
from control_flow.core.graph import EdgeType
from control_flow.core.task import Task, TaskStatus, get_tasks
from control_flow.utilities.context import ctx


def test_context_open_and_close():
    assert ctx.get("tasks") == []
    with Task("a") as ta:
        assert ctx.get("tasks") == [ta]
        with Task("b") as tb:
            assert ctx.get("tasks") == [ta, tb]
        assert ctx.get("tasks") == [ta]
    assert ctx.get("tasks") == []


def test_get_tasks_function():
    # assert get_tasks() == []
    with Task("a") as ta:
        assert get_tasks() == [ta]
        with Task("b") as tb:
            assert get_tasks() == [ta, tb]
        assert get_tasks() == [ta]
    assert get_tasks() == []


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
    assert task2 in task1.subtasks
    assert task2._parent == task1


def test_task_agent_assignment():
    agent = Agent(name="Test Agent")
    task = Task(objective="Test objective", agents=[agent])
    assert agent in task.agents


def test_task_context():
    with Flow():
        task = Task(objective="Test objective")
        assert task in Task._context_stack


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


def test_task_ready():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    assert not task2.is_ready()

    task1.mark_successful()
    assert task2.is_ready()


def test_task_hash():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2")
    assert hash(task1) != hash(task2)


def test_task_tools():
    task = Task(objective="Test objective")
    tools = task.get_tools()
    assert any(tool.name == f"mark_task_{task.id}_failed" for tool in tools)
    assert any(tool.name == f"mark_task_{task.id}_successful" for tool in tools)

    task.mark_successful()
    tools = task.get_tools()
    assert not any(tool.name == f"mark_task_{task.id}_failed" for tool in tools)
    assert not any(tool.name == f"mark_task_{task.id}_successful" for tool in tools)


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
