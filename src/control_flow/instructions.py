from contextlib import contextmanager
from typing import Generator, List

from control_flow.utilities.context import ctx
from control_flow.utilities.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def instructions(*instructions: str) -> Generator[list[str], None, None]:
    """
    Temporarily add instructions to the current instruction stack. The
    instruction is removed when the context is exited.

    with instructions("talk like a pirate"):
        ...

    """

    stack: list[str] = ctx.get("instructions", [])
    stack = stack + list(instructions)

    with ctx(instructions=stack):
        yield


def get_instructions() -> List[str]:
    """
    Get the current instruction stack.
    """
    stack = ctx.get("instructions", [])
    return stack
