from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar

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


@dataclass
class Edge:
    upstream: Task
    downstream: Task
    type: EdgeType

    def __repr__(self):
        return f"{self.type}: {self.upstream.friendly_name()} -> {self.downstream.friendly_name()}"

    def __hash__(self) -> id:
        return hash((id(self.upstream), id(self.downstream), self.type))


class Graph:
    def __init__(self, tasks: list[Task] = None, edges: list[Edge] = None):
        self.tasks: set[Task] = set()
        self.edges: set[Edge] = set()
        self._cache: dict[str[dict[Task, list[Task]]]] = {}
        if tasks:
            for task in tasks:
                self.add_task(task)
        if edges:
            for edge in edges:
                self.add_edge(edge)

    def add_task(self, task: Task):
        if task in self.tasks:
            return

        self.tasks.add(task)

        # add the task's parent
        if task.parent:
            self.add_edge(
                Edge(
                    upstream=task,
                    downstream=task.parent,
                    type=EdgeType.SUBTASK,
                )
            )

        # add the task's subtasks
        for subtask in task.subtasks:
            self.add_edge(
                Edge(
                    upstream=subtask,
                    downstream=task,
                    type=EdgeType.SUBTASK,
                )
            )

        # add the task's dependencies
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

    def upstream_tasks(
        self, start_tasks: list[Task], immediate: bool = False
    ) -> list[Task]:
        """
        Retrieve the upstream tasks (ancestors) of the given start tasks in topological order.

        Args:
            start_tasks (list[Task]): The list of starting tasks.
            immediate (bool): If True, only retrieve immediate upstream tasks.
                              If False, retrieve all upstream tasks recursively.

        Returns:
            list[Task]: A list of upstream tasks in topological order.
        """
        cache_key = (
            f"upstream_{'immediate' if immediate else 'all'}_{tuple(start_tasks)}"
        )
        if cache_key not in self._cache:
            result = set(start_tasks)
            visited = set()

            def _upstream(task):
                if task in visited:
                    return
                visited.add(task)
                for edge in self.upstream_edges().get(task, []):
                    if edge.upstream not in visited:
                        result.add(edge.upstream)
                        if not immediate:
                            _upstream(edge.upstream)

            for task in start_tasks:
                _upstream(task)

            # Perform a focused topological sort on the result
            sorted_tasks = self.topological_sort(list(result))
            self._cache[cache_key] = sorted_tasks

        return self._cache[cache_key]

    def downstream_tasks(
        self, start_tasks: list[Task], immediate: bool = False
    ) -> list[Task]:
        """
        Retrieve the downstream tasks (descendants) of the given start tasks in topological order.

        Args:
            start_tasks (list[Task]): The list of starting tasks.
            immediate (bool): If True, only retrieve immediate downstream tasks.
                              If False, retrieve all downstream tasks recursively.

        Returns:
            list[Task]: A list of downstream tasks in topological order.
        """
        cache_key = (
            f"downstream_{'immediate' if immediate else 'all'}_{tuple(start_tasks)}"
        )
        if cache_key not in self._cache:
            result = set(start_tasks)
            visited = set()

            def _downstream(task):
                if task in visited:
                    return
                visited.add(task)
                for edge in self.downstream_edges().get(task, []):
                    if edge.downstream not in visited:
                        result.add(edge.downstream)
                        if not immediate:
                            _downstream(edge.downstream)

            for task in start_tasks:
                _downstream(task)

            # Perform a focused topological sort on the result
            sorted_tasks = self.topological_sort(list(result))
            self._cache[cache_key] = sorted_tasks

        return self._cache[cache_key]

    def topological_sort(self, tasks: Optional[list[Task]] = None) -> list[Task]:
        """
        Perform a deterministic topological sort on the provided tasks or all tasks in the graph.

        Args:
            tasks (Optional[list[Task]]): A list of tasks to sort topologically.
                                        If None, all tasks in the graph are sorted.

        Returns:
            list[Task]: A list of tasks in topological order (upstream tasks first).
        """
        # Create a cache key based on the input tasks
        cache_key = (
            f"topo_sort_{tuple(sorted(id(task) for task in (tasks or self.tasks)))}"
        )

        # Check if the result is already in the cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        if tasks is None:
            tasks_to_sort = self.tasks
        else:
            tasks_to_sort = set(tasks)

        # Create a dictionary of tasks and their dependencies within tasks_to_sort
        dependencies = {task: set() for task in tasks_to_sort}
        for edge in self.edges:
            if edge.downstream in tasks_to_sort and edge.upstream in tasks_to_sort:
                dependencies[edge.downstream].add(edge.upstream)

        # Kahn's algorithm for topological sorting
        result = []
        no_incoming = [task for task in tasks_to_sort if not dependencies[task]]
        # sort to create a deterministic order
        no_incoming.sort(key=lambda t: t.created_at)

        while no_incoming:
            task = no_incoming.pop(0)
            result.append(task)

            # Remove the task from the dependencies of its neighbors
            for dependent_task in tasks_to_sort:
                if task in dependencies[dependent_task]:
                    dependencies[dependent_task].remove(task)
                    if not dependencies[dependent_task]:
                        no_incoming.append(dependent_task)
                        # resort to maintain deterministic order
                        no_incoming.sort(key=lambda t: t.created_at)

        # Check for cycles
        if len(result) != len(tasks_to_sort):
            raise ValueError(
                "The graph contains a cycle and cannot be topologically sorted"
            )

        # Cache the result before returning
        self._cache[cache_key] = result
        return result
