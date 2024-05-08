import math
from typing import Generator

import control_flow.core.task
from control_flow.core.task import Task


def any_incomplete(
    tasks: list[Task], max_iterations=None
) -> Generator[bool, None, None]:
    """
    An iterator that yields an iteration counter if its condition is met, and
    stops otherwise. Also stops if the max_iterations is reached.


    for loop_count in any_incomplete(tasks=[task1, task2], max_iterations=10):
        # will print 10 times if the tasks are still incomplete
        print(loop_count)

    """
    if max_iterations is None:
        max_iterations = math.inf

    i = 0
    while i < max_iterations:
        i += 1
        if control_flow.core.task.any_incomplete(tasks):
            yield i
        else:
            break
    return False
