from functools import partial

import pytest
from controlflow.agents import Agent, get_default_agent
from controlflow.controllers.graph import EdgeType
from controlflow.flows import Flow
from controlflow.instructions import instructions
from controlflow.tasks.task import Task, TaskStatus
from controlflow.utilities.context import ctx

SimpleTask = partial(Task, objective="test", result_type=None)


def test_context_open_and_close():
    assert ctx.get("tasks") == []
    with SimpleTask() as ta:
        assert ctx.get("tasks") == [ta]
        with SimpleTask() as tb:
            assert ctx.get("tasks") == [ta, tb]
        assert ctx.get("tasks") == [ta]
    assert ctx.get("tasks") == []


def test_task_requires_objective():
    with pytest.raises(ValueError):
        Task()


def test_task_initialization():
    task = Task(objective="Test objective")
    assert task.objective == "Test objective"
    assert task.status == TaskStatus.INCOMPLETE
    assert task.result is None
    assert task.error is None


def test_task_loads_instructions_at_creation():
    with instructions("test instruction"):
        task = SimpleTask()

    assert "test instruction" in task.instructions


def test_task_dependencies():
    task1 = SimpleTask()
    task2 = SimpleTask(depends_on=[task1])
    assert task1 in task2.depends_on
    assert task2 in task1._downstreams


def test_task_subtasks():
    task1 = SimpleTask()
    task2 = SimpleTask(parent=task1)
    assert task2 in task1._subtasks
    assert task2.parent is task1


def test_task_parent_context():
    with SimpleTask() as task1:
        with SimpleTask() as task2:
            task3 = SimpleTask()

    assert task3.parent is task2
    assert task2.parent is task1
    assert task1.parent is None

    assert task1._subtasks == {task2}
    assert task2._subtasks == {task3}
    assert task3._subtasks == set()


def test_task_agent_assignment():
    agent = Agent(name="Test Agent")
    task = SimpleTask(agents=[agent])
    assert agent in task.agents


def test_task_bad_agent_assignment():
    with pytest.raises(ValueError):
        SimpleTask(agents=[])


def test_task_loads_agent_from_parent():
    agent = Agent(name="Test Agent")
    with SimpleTask(agents=[agent]):
        child = SimpleTask()

    assert child.agents is None
    assert child.get_agents() == [agent]


def test_task_loads_agent_from_flow():
    def_agent = get_default_agent()
    agent = Agent(name="Test Agent")
    with Flow(agents=[agent]):
        task = SimpleTask()

        assert task.agents is None
        assert task.get_agents() == [agent]

    # outside the flow context, pick up the default agent
    assert task.get_agents() == [def_agent]


def test_task_loads_agent_from_default_if_none_otherwise():
    agent = get_default_agent()
    task = SimpleTask()

    assert task.agents is None
    assert task.get_agents() == [agent]


def test_task_loads_agent_from_parent_before_flow():
    agent1 = Agent(name="Test Agent 1")
    agent2 = Agent(name="Test Agent 2")
    with Flow(agents=[agent1]):
        with SimpleTask(agents=[agent2]):
            child = SimpleTask()

    assert child.agents is None
    assert child.get_agents() == [agent2]


def test_task_tracking():
    with Flow() as flow:
        task = SimpleTask()
        assert task in flow.tasks.values()


def test_task_tracking_on_call():
    task = SimpleTask()
    with Flow() as flow:
        task.run_once()
    assert task in flow.tasks.values()


class TestTaskStatus:
    def test_task_status_transitions(self):
        task = SimpleTask()
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

        task = SimpleTask()
        task.mark_failed()
        assert not task.is_incomplete()
        assert task.is_complete()
        assert not task.is_successful()
        assert task.is_failed()
        assert not task.is_skipped()

        task = SimpleTask()
        task.mark_skipped()
        assert not task.is_incomplete()
        assert task.is_complete()
        assert not task.is_successful()
        assert not task.is_failed()
        assert task.is_skipped()

    def test_validate_upstream_dependencies_on_success(self):
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        with pytest.raises(ValueError, match="cannot be marked successful"):
            task2.mark_successful()
        task1.mark_successful()
        task2.mark_successful()

    def test_validate_subtask_dependencies_on_success(self):
        task1 = SimpleTask()
        task2 = SimpleTask(parent=task1)
        with pytest.raises(ValueError, match="cannot be marked successful"):
            task1.mark_successful()
        task2.mark_successful()
        task1.mark_successful()

    def test_task_ready(self):
        task1 = SimpleTask()
        assert task1.is_ready()

    def test_task_not_ready_if_successful(self):
        task1 = SimpleTask()
        task1.mark_successful()
        assert not task1.is_ready()

    def test_task_not_ready_if_failed(self):
        task1 = SimpleTask()
        task1.mark_failed()
        assert not task1.is_ready()

    def test_task_not_ready_if_dependencies_are_ready(self):
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        assert task1.is_ready()
        assert not task2.is_ready()

    def test_task_ready_if_dependencies_are_ready(self):
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        task1.mark_successful()
        assert not task1.is_ready()
        assert task2.is_ready()

    def test_task_hash(self):
        task1 = SimpleTask()
        task2 = SimpleTask()
        assert hash(task1) != hash(task2)

    def test_ready_task_adds_tools(self):
        task = SimpleTask()
        assert task.is_ready()

        tools = task.get_tools()
        assert any(tool.name == f"mark_task_{task.id}_failed" for tool in tools)
        assert any(tool.name == f"mark_task_{task.id}_successful" for tool in tools)

    def test_completed_task_does_not_add_tools(self):
        task = SimpleTask()
        task.mark_successful()
        tools = task.get_tools()
        assert not any(tool.name == f"mark_task_{task.id}_failed" for tool in tools)
        assert not any(tool.name == f"mark_task_{task.id}_successful" for tool in tools)

    def test_task_with_incomplete_upstream_does_not_add_tools(self):
        upstream_task = SimpleTask()
        downstream_task = SimpleTask(depends_on=[upstream_task])
        tools = downstream_task.get_tools()
        assert not any(
            tool.name == f"mark_task_{downstream_task.id}_failed" for tool in tools
        )
        assert not any(
            tool.name == f"mark_task_{downstream_task.id}_successful" for tool in tools
        )

    def test_task_with_incomplete_subtask_does_not_add_tools(self):
        parent = SimpleTask()
        SimpleTask(parent=parent)
        tools = parent.get_tools()
        assert not any(tool.name == f"mark_task_{parent.id}_failed" for tool in tools)
        assert not any(
            tool.name == f"mark_task_{parent.id}_successful" for tool in tools
        )


class TestTaskToGraph:
    def test_single_task_graph(self):
        task = SimpleTask()
        graph = task.as_graph()
        assert len(graph.tasks) == 1
        assert task in graph.tasks
        assert len(graph.edges) == 0

    def test_task_with_subtasks_graph(self):
        task1 = SimpleTask()
        task2 = SimpleTask(parent=task1)
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
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
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
        task1 = SimpleTask()
        task2 = SimpleTask(depends_on=[task1])
        task3 = SimpleTask(objective="Task 3", parent=task2)
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
