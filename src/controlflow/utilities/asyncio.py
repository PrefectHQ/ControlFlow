import asyncio
import functools
from typing import Any, Callable, Coroutine, TypeVar, cast

from prefect.utilities.asyncutils import run_sync

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


class ExposeSyncMethodsMixin:
    """
    A mixin that can take functions decorated with `expose_sync_method`
    and automatically create synchronous versions.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for method in list(cls.__dict__.values()):
            if callable(method) and hasattr(method, "_sync_name"):
                sync_method_name = method._sync_name
                setattr(cls, sync_method_name, method._sync_wrapper)


def expose_sync_method(name: str) -> Callable[..., Any]:
    """
    Decorator that automatically exposes synchronous versions of async methods.
    Note it doesn't work with classmethods.

    Args:
        name: The name of the synchronous method.

    Returns:
        The decorated function.

    Example:
        Basic usage:
        ```python
        class MyClass(ExposeSyncMethodsMixin):

            @expose_sync_method("my_method")
            async def my_method_async(self):
                return 42

        my_instance = MyClass()
        await my_instance.my_method_async() # returns 42
        my_instance.my_method()  # returns 42
        ```
    """

    def decorator(
        async_method: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(async_method)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            coro = async_method(*args, **kwargs)
            return run_sync(coro)

        # Cast the sync_wrapper to the same type as the async_method to give the
        # type checker the needed information.
        casted_sync_wrapper = cast(Callable[..., T], sync_wrapper)

        # Attach attributes to the async wrapper
        setattr(async_method, "_sync_wrapper", casted_sync_wrapper)
        setattr(async_method, "_sync_name", name)

        # return the original async method; the sync wrapper will be added to
        # the class by the init hook
        return async_method

    return decorator
