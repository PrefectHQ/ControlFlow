import inspect
from contextlib import contextmanager
from typing import Generator, List

from marvin.utilities.logging import get_logger

from control_flow.context import ctx
from control_flow.flow import AIFlow

logger = get_logger(__name__)


@contextmanager
def instructions(
    *instructions: str,
    post_add_message: bool = False,
    post_remove_message: bool = False,
) -> Generator[list[str], None, None]:
    """
    Temporarily add instructions to the current instruction stack. The
    instruction is removed when the context is exited.

    If `post_add_message` is True, a message will be added to the flow when the
    instruction is added. If `post_remove_message` is True, a message will be
    added to the flow when the instruction is removed. These explicit reminders
    can help when agents infer instructions more from history.

    with instructions("talk like a pirate"):
        ...

    """

    if post_add_message or post_remove_message:
        flow: AIFlow = ctx.get("flow")
        if flow is None:
            raise ValueError(
                "instructions() with message posting must be used within a flow context"
            )

    stack: list[str] = ctx.get("instructions", [])
    stack = stack + list(instructions)

    with ctx(instructions=stack):
        try:
            if post_add_message:
                for instruction in instructions:
                    flow.add_message(
                        inspect.cleandoc(
                            """
                            # SYSTEM MESSAGE: INSTRUCTION ADDED

                            The following instruction is now active:                    
                            
                            <instruction>
                            {instruction}
                            </instruction>
                            
                            Always consult your current instructions before acting.
                            """
                        ).format(instruction=instruction)
                    )
            yield

            # yield new_stack
        finally:
            if post_remove_message:
                for instruction in instructions:
                    flow.add_message(
                        inspect.cleandoc(
                            """
                            # SYSTEM MESSAGE: INSTRUCTION REMOVED

                            The following instruction is no longer active:                    
                            
                            <instruction>
                            {instruction}
                            </instruction>
                            
                            Always consult your current instructions before acting.
                            """
                        ).format(instruction=instruction)
                    )


def get_instructions() -> List[str]:
    """
    Get the current instruction stack.
    """
    stack = ctx.get("instructions", [])
    return stack
