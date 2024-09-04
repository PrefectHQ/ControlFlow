import logging
from functools import lru_cache
from typing import Optional

import controlflow


@lru_cache()
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieves a logger with the given name, or the root logger if no name is given.

    Args:
        name: The name of the logger to retrieve.

    Returns:
        The logger with the given name, or the root logger if no name is given.

    Example:
        Basic Usage of `get_logger`
        ```python
        from controlflow.utilities.logging import get_logger

        logger = get_logger("controlflow.test")
        logger.info("This is a test") # Output: controlflow.test: This is a test

        debug_logger = get_logger("controlflow.debug")
        debug_logger.debug_kv("TITLE", "log message", "green")
        ```
    """
    parent_logger = logging.getLogger("controlflow")

    if name:
        # Append the name if given but allow explicit full names e.g. "controlflow.test"
        # should not become "controlflow.controlflow.test"
        if not name.startswith(parent_logger.name + "."):
            logger = parent_logger.getChild(name)
        else:
            logger = logging.getLogger(name)
    else:
        logger = parent_logger

    return logger


def setup_logging(level: Optional[str] = None) -> None:
    logger = get_logger()

    if level is not None:
        logger.setLevel(level)
    else:
        logger.setLevel(controlflow.settings.log_level)


def deprecated(message: str, version: str):
    """
    Decorator to mark a function as deprecated.

    Args:
        message (str): The deprecation message.
        version (str): The version in which the function is deprecated.

    Returns:
        function: The decorated function.

    Example:
        @deprecated("This function is deprecated", "1.0")
        def my_function():
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            get_logger(__file__).warn(
                f"WARNING: {func.__name__} is deprecated as of version {version}. {message}".strip(),
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
