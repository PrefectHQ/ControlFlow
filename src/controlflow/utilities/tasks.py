from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
)

from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.tasks.task import Task

T = TypeVar("T")
logger = get_logger(__name__)


def visit_task_collection(
    val: Any, visitor: Callable, recursion_limit: int = 10, _counter: int = 0
) -> list["Task"]:
    """
    Recursively visits a task collection and applies a visitor function to each task.

    Args:
        val (Any): The task collection to visit.
        visitor (Callable): The visitor function to apply to each task.
        recursion_limit (int, optional): The maximum recursion limit. Defaults to 3.
        _counter (int, optional): Internal counter to track recursion depth. Defaults to 0.

    Returns:
        list["Task"]: The modified task collection after applying the visitor function.

    """
    from controlflow.tasks.task import Task

    if _counter >= recursion_limit:
        return val

    if isinstance(val, dict):
        result = {}
        for key, value in list(val.items()):
            result[key] = visit_task_collection(
                value,
                visitor=visitor,
                recursion_limit=recursion_limit,
                _counter=_counter + 1,
            )
        return result
    elif isinstance(val, (list, set, tuple)):
        result = []
        for item in val:
            result.append(
                visit_task_collection(
                    item,
                    visitor=visitor,
                    recursion_limit=recursion_limit,
                    _counter=_counter + 1,
                )
            )
        return type(val)(result)
    elif isinstance(val, Task):
        return visitor(val)

    return val


def collect_tasks(val: T) -> list["Task"]:
    """
    Given a collection of tasks, returns a list of all tasks in the collection.
    """

    tasks = []

    def visit_task(task: "Task"):
        tasks.append(task)
        return task

    visit_task_collection(val, visit_task)
    return tasks


def resolve_tasks(val: T) -> T:
    """
    Given a collection of tasks, runs them to completion and returns the results.
    """

    def visit_task(task: "Task"):
        return task.run()

    return visit_task_collection(val, visit_task)


def any_incomplete(tasks: list["Task"]) -> bool:
    return any(t.is_incomplete() for t in tasks)


def all_complete(tasks: list["Task"]) -> bool:
    return all(t.is_complete() for t in tasks)


def all_successful(tasks: list["Task"]) -> bool:
    return all(t.is_successful() for t in tasks)


def any_failed(tasks: list["Task"]) -> bool:
    return any(t.is_failed() for t in tasks)


def none_failed(tasks: list["Task"]) -> bool:
    return not any_failed(tasks)
