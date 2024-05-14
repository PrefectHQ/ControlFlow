from enum import Enum

from pydantic import BaseModel

from controlflow.core.task import Task


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
        for subtask in task.subtasks:
            self.add_edge(
                Edge(
                    upstream=subtask,
                    downstream=task,
                    type=EdgeType.SUBTASK,
                )
            )

        for upstream in task.depends_on:
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

    def upstream_dependencies(
        self,
        tasks: list[Task],
        prune_completed: bool = True,
        include_tasks: bool = False,
    ) -> list[Task]:
        """
        From a list of tasks, returns the subgraph of tasks that are directly or
        indirectly dependencies of those tasks. A dependency means following
        upstream edges, so it includes tasks that are considered explicit
        dependencies as well as any subtasks that are considered implicit
        dependencies.

        If `prune_completed` is True, the subgraph will be pruned to stop
        traversal after adding any completed tasks.

        If `include_tasks` is True, the subgraph will include the tasks provided.
        """
        subgraph = set()
        upstreams = self.upstream_edges()
        # copy stack to allow difference update with original tasks
        stack = [t for t in tasks]
        while stack:
            current = stack.pop()
            if current in subgraph:
                continue

            subgraph.add(current)
            # if prune_completed, stop traversal if the current task is complete
            if prune_completed and current.is_complete():
                continue
            stack.extend([edge.upstream for edge in upstreams[current]])

        if not include_tasks:
            subgraph.difference_update(tasks)
        return list(subgraph)

    def ready_tasks(self, tasks: list[Task] = None) -> list[Task]:
        """
        Returns a list of tasks that are ready to run, meaning that all of their
        dependencies have been completed. If `tasks` is provided, only tasks in
        the upstream dependency subgraph of those tasks will be considered.

        Ready tasks will be returned in the order they were created.
        """
        if tasks is None:
            candidates = self.tasks
        else:
            candidates = self.upstream_dependencies(tasks, include_tasks=True)
        return sorted(
            [task for task in candidates if task.is_ready()], key=lambda t: t.created_at
        )
