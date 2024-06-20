from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

from controlflow.tasks.task import Task

T = TypeVar("T")


class EdgeType(Enum):
    """
    Edges represent the relationship between two tasks in a graph.

    - `DEPENDENCY_OF` means that the downstream task depends on the upstream task.
    - `PARENT` means that the downstream task is the parent of the upstream task.

    Example:

    # write paper
        ## write outline
        ## write draft based on outline

    Edges:
    outline -> paper # SUBTASK (outline is a subtask of paper)
    draft -> paper # SUBTASK (draft is a subtask of paper)
    outline -> draft # DEPENDENCY (outline is a dependency of draft)

    """

    DEPENDENCY = "dependency"
    SUBTASK = "subtask"


class Edge(BaseModel):
    upstream: Task
    downstream: Task
    type: EdgeType

    def __repr__(self):
        return f"{self.type}: {self.upstream.friendly_name()} -> {self.downstream.friendly_name()}"

    def __hash__(self) -> int:
        return id(self)


class Graph(BaseModel):
    tasks: set[Task] = set()
    edges: set[Edge] = set()
    _cache: dict[str, dict[Task, list[Task]]] = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def from_tasks(cls, tasks: list[Task]) -> "Graph":
        graph = cls()
        for task in tasks:
            graph.add_task(task)
        return graph

    def add_task(self, task: Task):
        if task in self.tasks:
            return
        self.tasks.add(task)
        for subtask in task._subtasks:
            self.add_edge(
                Edge(
                    upstream=subtask,
                    downstream=task,
                    type=EdgeType.SUBTASK,
                )
            )

        for upstream in task.depends_on:
            if upstream not in task._subtasks:
                self.add_edge(
                    Edge(
                        upstream=upstream,
                        downstream=task,
                        type=EdgeType.DEPENDENCY,
                    )
                )
        self._cache.clear()

    def add_edge(self, edge: Edge):
        if edge in self.edges:
            return
        self.edges.add(edge)
        self.add_task(edge.upstream)
        self.add_task(edge.downstream)
        self._cache.clear()

    def upstream_edges(self) -> dict[Task, list[Edge]]:
        if "upstream_edges" not in self._cache:
            graph = {}
            for task in self.tasks:
                graph[task] = []
            for edge in self.edges:
                graph[edge.downstream].append(edge)
            self._cache["upstream_edges"] = graph
        return self._cache["upstream_edges"]

    def downstream_edges(self) -> dict[Task, list[Edge]]:
        if "downstream_edges" not in self._cache:
            graph = {}
            for task in self.tasks:
                graph[task] = []
            for edge in self.edges:
                graph[edge.upstream].append(edge)
            self._cache["downstream_edges"] = graph
        return self._cache["downstream_edges"]

    def topological_sort(self) -> list[Task]:
        """
        Perform a topological sort on the graph and return the sorted tasks.

        This is a depth-first search algorithm that visits each node and its
        upstream dependencies. This maintains context as much as possible
        when traversing the graph (e.g. all else equal, the graph will visit
        as many dependent tasks in a row before jumping to a "new" branch).

        Returns:
            list: A list of tasks in the order of their dependencies.
        """
        if "topological_sort" not in self._cache:
            visited = set()
            stack = []

            dependencies = self.upstream_edges()
            created_at = {task: task.created_at for task in self.tasks}

            def dfs(task):
                visited.add(task)
                for dependent in sorted(
                    dependencies.get(task, []), key=lambda x: created_at[x.upstream]
                ):
                    if dependent.upstream not in visited:
                        dfs(dependent.upstream)
                stack.append(task)

            all_tasks = self.tasks
            for task in sorted(all_tasks, key=lambda x: created_at[x]):
                if task not in visited:
                    dfs(task)

            self._cache["topological_sort"] = stack
        return self._cache["topological_sort"]
