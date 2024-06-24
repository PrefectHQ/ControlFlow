# test_graph.py
from controlflow.controllers.graph import Edge, EdgeType, Graph
from controlflow.tasks.task import Task


def test_graph_initialization():
    graph = Graph()
    assert len(graph.tasks) == 0
    assert len(graph.edges) == 0


def test_add_task():
    graph = Graph()
    task = Task(objective="Test objective")
    graph.add_task(task)
    assert len(graph.tasks) == 1
    assert task in graph.tasks


def test_add_edge():
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


def test_from_tasks():
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


def test_upstream_edges():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    graph = Graph.from_tasks([task1, task2])
    upstream_edges = graph.upstream_edges()
    assert len(upstream_edges[task1]) == 0
    assert len(upstream_edges[task2]) == 1
    assert upstream_edges[task2][0].upstream == task1


def test_downstream_edges():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    graph = Graph.from_tasks([task1, task2])
    downstream_edges = graph.downstream_edges()
    assert len(downstream_edges[task1]) == 1
    assert len(downstream_edges[task2]) == 0
    assert downstream_edges[task1][0].downstream == task2


def test_topological_sort():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2", depends_on=[task1])
    task3 = Task(objective="Task 3", depends_on=[task2])
    task4 = Task(objective="Task 4", depends_on=[task3])
    graph = Graph.from_tasks([task1, task2, task3, task4])
    sorted_tasks = graph.topological_sort()
    assert len(sorted_tasks) == 4
    assert sorted_tasks.index(task1) < sorted_tasks.index(task2)
    assert sorted_tasks.index(task2) < sorted_tasks.index(task3)
    assert sorted_tasks.index(task3) < sorted_tasks.index(task4)


def test_topological_sort_with_fan_in_and_fan_out():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2")
    task3 = Task(objective="Task 3")

    edge1 = Edge(upstream=task1, downstream=task2, type=EdgeType.DEPENDENCY)
    edge2 = Edge(upstream=task1, downstream=task3, type=EdgeType.DEPENDENCY)
    edge3 = Edge(upstream=task2, downstream=task3, type=EdgeType.DEPENDENCY)

    graph = Graph()
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)

    sorted_tasks = graph.topological_sort()

    assert len(sorted_tasks) == 3
    assert sorted_tasks.index(task1) < sorted_tasks.index(task2)
    assert sorted_tasks.index(task1) < sorted_tasks.index(task3)
    assert sorted_tasks.index(task2) < sorted_tasks.index(task3)

    assert graph.topological_sort(tasks=[task3, task1]) == [task1, task3]


def test_upstream_tasks():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2")
    task3 = Task(objective="Task 3")

    edge1 = Edge(upstream=task1, downstream=task2, type=EdgeType.DEPENDENCY)
    edge2 = Edge(upstream=task1, downstream=task3, type=EdgeType.DEPENDENCY)
    edge3 = Edge(upstream=task2, downstream=task3, type=EdgeType.DEPENDENCY)

    graph = Graph()
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)

    assert graph.upstream_tasks([task3]) == [task1, task2]
    assert graph.upstream_tasks([task2]) == [task1]
    assert graph.upstream_tasks([task1]) == []

    # never include a start task in the usptream list
    assert graph.upstream_tasks([task1, task3]) == [task2]


def test_downstream_tasks():
    task1 = Task(objective="Task 1")
    task2 = Task(objective="Task 2")
    task3 = Task(objective="Task 3")

    edge1 = Edge(upstream=task1, downstream=task2, type=EdgeType.DEPENDENCY)
    edge2 = Edge(upstream=task1, downstream=task3, type=EdgeType.DEPENDENCY)
    edge3 = Edge(upstream=task2, downstream=task3, type=EdgeType.DEPENDENCY)

    graph = Graph()
    graph.add_edge(edge1)
    graph.add_edge(edge2)
    graph.add_edge(edge3)

    assert graph.downstream_tasks([task3]) == []
    assert graph.downstream_tasks([task2]) == [task3]
    assert graph.downstream_tasks([task1]) == [task2, task3]

    # never include a start task in the downstream list
    assert graph.downstream_tasks([task1, task3]) == [task2]
