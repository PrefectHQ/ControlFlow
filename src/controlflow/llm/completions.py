import inspect
import math
from typing import AsyncGenerator, Callable, Generator, Tuple, Union

import litellm
from litellm.utils import trim_messages

import controlflow
from controlflow.llm.handlers import AsyncStreamHandler, StreamHandler
from controlflow.llm.tools import (
    as_tools,
    get_tool_calls,
    handle_tool_call,
)
from controlflow.utilities.types import (
    ControlFlowMessage,
    as_cf_messages,
    as_oai_messages,
)


async def maybe_coro(coro):
    if inspect.isawaitable(coro):
        await coro


def completion(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    max_iterations=None,
    handlers: list[StreamHandler] = None,
    **kwargs,
) -> list[ControlFlowMessage]:
    """
    Perform completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Returns:
        A list of ControlFlowMessage objects representing the completion response.
    """
    response_messages = []
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    counter = 0
    while not response_messages or get_tool_calls(response_messages):
        completion_messages = trim_messages(
            as_oai_messages(messages + new_messages), model=model
        )
        response = litellm.completion(
            model=model,
            messages=completion_messages,
            tools=[t.model_dump() for t in tools] if tools else None,
            **kwargs,
        )

        response_messages = as_cf_messages([response])

        # on message done
        for h in handlers:
            for msg in response_messages:
                if msg.has_tool_calls():
                    h.on_tool_call_done(msg)
                else:
                    h.on_message_done(msg)

        new_messages.extend(response_messages)
        for tool_call in get_tool_calls(response_messages):
            tool_message = handle_tool_call(tool_call, tools)

            # on tool result
            for h in handlers:
                h.on_tool_result(tool_message)
            new_messages.append(tool_message)

        counter += 1
        if counter >= (max_iterations or math.inf):
            break

    return new_messages


def completion_stream(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    max_iterations: int = None,
    handlers: list[StreamHandler] = None,
    **kwargs,
) -> Generator[Tuple[litellm.ModelResponse, litellm.ModelResponse], None, None]:
    """
    Perform streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Yields:
        Each message

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """

    snapshot_message = None
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    counter = 0
    while not snapshot_message or get_tool_calls([snapshot_message]):
        completion_messages = trim_messages(
            as_oai_messages(messages + new_messages), model=model
        )
        response = litellm.completion(
            model=model,
            messages=completion_messages,
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=True,
            **kwargs,
        )

        deltas = []
        for delta in response:
            deltas.append(delta)
            snapshot = litellm.stream_chunk_builder(deltas)
            delta_message, snapshot_message = as_cf_messages([delta, snapshot])

            # on message created
            if len(deltas) == 1:
                for h in handlers:
                    if snapshot_message.has_tool_calls():
                        h.on_tool_call_created(delta=delta_message)
                    else:
                        h.on_message_created(delta=delta_message)

            # on message delta
            for h in handlers:
                if snapshot_message.has_tool_calls():
                    h.on_tool_call_delta(delta=delta_message, snapshot=snapshot_message)
                else:
                    h.on_message_delta(delta=delta_message, snapshot=snapshot_message)

        yield snapshot_message

        new_messages.append(snapshot_message)

        # on message done
        for h in handlers:
            if snapshot_message.has_tool_calls():
                h.on_tool_call_done(snapshot_message)
            else:
                h.on_message_done(snapshot_message)

        # tool calls
        for tool_call in get_tool_calls([snapshot_message]):
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                h.on_tool_result(tool_message)
            new_messages.append(tool_message)
            yield tool_message

        counter += 1
        if counter >= (max_iterations or math.inf):
            break


async def completion_async(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    max_iterations=None,
    handlers: list[Union[AsyncStreamHandler, StreamHandler]] = None,
    **kwargs,
) -> list[ControlFlowMessage]:
    """
    Perform asynchronous completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Returns:
        A list of ControlFlowMessage objects representing the completion response.
    """
    response_messages = []
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    counter = 0
    while not response_messages or get_tool_calls(response_messages):
        completion_messages = trim_messages(
            as_oai_messages(messages + new_messages), model=model
        )
        response = await litellm.acompletion(
            model=model,
            messages=completion_messages,
            tools=[t.model_dump() for t in tools] if tools else None,
            **kwargs,
        )

        response_messages = as_cf_messages([response])

        # on message done
        for h in handlers:
            for msg in response_messages:
                if msg.has_tool_calls():
                    await maybe_coro(h.on_tool_call_done(msg))
                else:
                    await maybe_coro(h.on_message_done(msg))

        new_messages.extend(response_messages)
        for tool_call in get_tool_calls(response_messages):
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                await maybe_coro(h.on_tool_result(tool_message))
            new_messages.append(tool_message)

        counter += 1
        if counter >= (max_iterations or math.inf):
            break

    return new_messages


async def completion_stream_async(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    max_iterations: int = None,
    handlers: list[Union[AsyncStreamHandler, StreamHandler]] = None,
    **kwargs,
) -> AsyncGenerator[ControlFlowMessage, None]:
    """
    Perform asynchronous streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Yields:
        Each message

    Returns:
        The final completion response as a list of ControlFlowMessage objects.
    """

    snapshot_message = None
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    counter = 0
    while not snapshot_message or get_tool_calls([snapshot_message]):
        completion_messages = trim_messages(
            as_oai_messages(messages + new_messages), model=model
        )
        response = await litellm.acompletion(
            model=model,
            messages=completion_messages,
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=True,
            **kwargs,
        )

        deltas = []
        async for delta in response:
            deltas.append(delta)
            snapshot = litellm.stream_chunk_builder(deltas)
            delta_message, snapshot_message = as_cf_messages([delta, snapshot])

            # on message created
            if len(deltas) == 1:
                for h in handlers:
                    if snapshot_message.has_tool_calls():
                        await maybe_coro(h.on_tool_call_created(delta=delta_message))
                    else:
                        await maybe_coro(h.on_message_created(delta=delta_message))

            # on message delta
            for h in handlers:
                if snapshot_message.has_tool_calls():
                    await maybe_coro(
                        h.on_tool_call_delta(
                            delta=delta_message, snapshot=snapshot_message
                        )
                    )
                else:
                    await maybe_coro(
                        h.on_message_delta(
                            delta=delta_message, snapshot=snapshot_message
                        )
                    )

        yield snapshot_message

        new_messages.append(snapshot_message)

        # on message done
        for h in handlers:
            if snapshot_message.has_tool_calls():
                await maybe_coro(h.on_tool_call_done(snapshot_message))
            else:
                await maybe_coro(h.on_message_done(snapshot_message))

        # tool calls
        for tool_call in get_tool_calls([snapshot_message]):
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                await maybe_coro(h.on_tool_result(tool_message))
            new_messages.append(tool_message)
            yield tool_message

        counter += 1
        if counter >= (max_iterations or math.inf):
            break
