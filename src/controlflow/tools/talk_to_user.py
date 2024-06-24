import asyncio
import contextlib
from typing import TYPE_CHECKING

from prefect.context import FlowRunContext
from prefect.input.run_input import receive_input
from rich.prompt import Prompt as RichPrompt

import controlflow
from controlflow.tools import tool

if TYPE_CHECKING:
    pass


class Prompt(RichPrompt):
    # remove the prompt suffix
    prompt_suffix = " "


async def get_terminal_input(message: str):
    # as a convenience, we wait for human input on the local terminal
    # this is not necessary for the flow to run, but can be useful for testing
    loop = asyncio.get_event_loop()
    user_input = await loop.run_in_executor(
        None,
        RichPrompt.ask,
        f"\n[bold blue]🤖 Agent:[/] [blue]{message}[/]\nType your response",
    )
    return user_input


# async def get_tui_input(tui: "TUIApp", message: str):
#     container = []
#     await tui.get_input(message=message, container=container)
#     while not container:
#         await asyncio.sleep(0.1)
#     return container[0]


async def get_flow_run_input(message: str):
    async for response in receive_input(
        str, flow_run_id=FlowRunContext.get().flow_run.id, poll_interval=0.2
    ):
        return response


@tool
async def talk_to_user(message: str, get_response: bool = True) -> str:
    """
    Send a message to the human user and optionally wait for a response. If
    `get_response` is True, the function will return the user's response,
    otherwise it will return a simple confirmation. Do not send the user
    concurrent messages that require responses, as this will cause confusion.

    You may need to ask the human about multiple tasks at once. Consolidate your
    questions into a single message. For example, if Task 1 requires information
    X and Task 2 needs information Y, send a single message that naturally asks
    for both X and Y.
    """

    if get_response:
        tasks = []
        # if running in a Prefect flow, listen for a remote input
        if (frc := FlowRunContext.get()) and frc.flow_run and frc.flow_run.id:
            remote_input = asyncio.create_task(get_flow_run_input(message=message))
            tasks.append(remote_input)
        # if terminal input is enabled, listen for local input
        if controlflow.settings.enable_local_input:
            local_input = asyncio.create_task(get_terminal_input(message=message))
            tasks.append(local_input)
        if not tasks:
            raise ValueError(
                "No input sources enabled. Either run this task in a flow or enable local input in settings."
            )

        # wait for either the terminal input or the API response
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Get the result of the first completed task
        result = done.pop().result()

        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        return f"User response: {result}"

    return "Message sent to user."
