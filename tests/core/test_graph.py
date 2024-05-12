# test_graph.py
from control_flow.core.graph import Edge, EdgeType, Graph
from control_flow.core.task import Task


class TestGraph:
    def test_graph_initialization(self):
        graph = Graph()
        assert len(graph.tasks) == 0
        assert len(graph.edges) == 0

    def test_add_task(self):
        graph = Graph()
        task = Task(objective="Test objective")
        graph.add_task(task)
        assert len(graph.tasks) == 1
        assert task in graph.tasks

    def test_add_edge(self):
        graph = Graph()
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2")
        edge = Edge(upstream=task1, downstream=task2, type=EdgeType.DEPENDENCY)
        graph.add_edge(edge)
        assert len(graph.tasks) == 2
        assert task1 in graph.tasks
        assert task2 in graph.tasks
        assert len(graph.edges) == 1
        assert edge in graph.edges

    def test_from_tasks(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
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

    def test_upstream_edges(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        graph = Graph.from_tasks([task1, task2])
        upstream_edges = graph.upstream_edges()
        assert len(upstream_edges[task1]) == 0
        assert len(upstream_edges[task2]) == 1
        assert upstream_edges[task2][0].upstream == task1

    def test_downstream_edges(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        graph = Graph.from_tasks([task1, task2])
        downstream_edges = graph.downstream_edges()
        assert len(downstream_edges[task1]) == 1
        assert len(downstream_edges[task2]) == 0
        assert downstream_edges[task1][0].downstream == task2

    def test_upstream_dependencies(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
        dependencies = graph.upstream_dependencies([task3])
        assert len(dependencies) == 3
        assert task1 in dependencies
        assert task2 in dependencies
        assert task3 in dependencies

    def test_ready_tasks(self):
        task1 = Task(objective="Task 1")
        task2 = Task(objective="Task 2", depends_on=[task1])
        task3 = Task(objective="Task 3", parent=task2)
        graph = Graph.from_tasks([task1, task2, task3])
        ready_tasks = graph.ready_tasks()
        assert len(ready_tasks) == 1
        assert task1 in ready_tasks

        task1.mark_successful()
        ready_tasks = graph.ready_tasks()
        assert len(ready_tasks) == 1
        assert task2 in ready_tasks

        task2.mark_successful()
        ready_tasks = graph.ready_tasks()
        assert len(ready_tasks) == 1
        assert task3 in ready_tasks
