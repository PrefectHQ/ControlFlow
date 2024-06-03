import asyncio
from typing import TypeVar

T = TypeVar("T")

BACKGROUND_TASKS = set()


def create_task(coro):
    """
    Creates async background tasks in a way that is safe from garbage
    collection.

    See
    https://textual.textualize.io/blog/2023/02/11/the-heisenbug-lurking-in-your-async-code/

    Example:

    async def my_coro(x: int) -> int:
        return x + 1

    # safely submits my_coro for background execution
    create_task(my_coro(1))
    """  # noqa: E501
    task = asyncio.create_task(coro)
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return task
