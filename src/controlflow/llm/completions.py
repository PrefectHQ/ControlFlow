import inspect
import math
from typing import AsyncGenerator, Callable, Generator, Optional, Tuple, Union

import litellm
from litellm.utils import trim_messages
from pydantic import field_validator

import controlflow
from controlflow.llm.handlers import AsyncStreamHandler, StreamHandler
from controlflow.llm.tools import (
    as_tools,
    get_tool_calls,
    handle_tool_call,
    has_tool_calls,
)
from controlflow.utilities.types import ControlFlowModel, Message, ToolResult


def as_cf_message(message: Union[Message, litellm.Message]) -> Message:
    if isinstance(message, Message):
        return message
    return Message(**message.model_dump())


async def maybe_coro(coro):
    if inspect.isawaitable(coro):
        await coro


class Response(ControlFlowModel):
    messages: list[Message] = []
    responses: list[litellm.ModelResponse] = []

    @field_validator("messages", mode="before")
    def _validate_messages(cls, v):
        return [as_cf_message(m) for m in v]

    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def last_response(self) -> Optional[litellm.ModelResponse]:
        return self.responses[-1] if self.responses else None

    def tool_calls(self) -> list[ToolResult]:
        return [
            m["_tool_call"]
            for m in self.messages
            if m.role == "tool" and m.get("_tool_call") is not None
        ]


def completion(
    messages: list[Union[dict, Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations=None,
    handlers: list[StreamHandler] = None,
    **kwargs,
) -> Response:
    """
    Perform completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        call_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Returns:
        A Response object representing the completion response.
    """

    response = None
    responses = []
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    while not response or has_tool_calls(response):
        response = litellm.completion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools] if tools else None,
            **kwargs,
        )

        responses.append(response)

        # on message done
        for h in handlers:
            h.on_message_done(response)
        new_messages.append(response.choices[0].message)

        for tool_call in get_tool_calls(response):
            for h in handlers:
                h.on_tool_call_done(tool_call=tool_call)
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                h.on_tool_result(tool_message)
            new_messages.append(tool_message)

        if len(responses) >= (max_iterations or math.inf):
            break

    return Response(
        messages=new_messages,
        responses=responses,
    )


def completion_stream(
    messages: list[Union[dict, Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations: int = None,
    handlers: list[StreamHandler] = None,
    response_callback: Callable[[Response], None] = None,
    **kwargs,
) -> Generator[Tuple[litellm.ModelResponse, litellm.ModelResponse], None, None]:
    """
    Perform streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        call_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Yields:
        A tuple containing the current completion delta and the snapshot of the completion response.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """

    response = None
    responses = []
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    while not response or has_tool_calls(response):
        deltas = []
        is_tool_call = False
        for delta in litellm.completion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=True,
            **kwargs,
        ):
            deltas.append(delta)
            response = litellm.stream_chunk_builder(deltas)

            # on message created
            if len(deltas) == 1:
                if get_tool_calls(response):
                    is_tool_call = True
                for h in handlers:
                    if is_tool_call:
                        h.on_tool_call_created(delta=delta)
                    else:
                        h.on_message_created(delta=delta)

            # on message delta
            for h in handlers:
                if is_tool_call:
                    h.on_tool_call_delta(delta=delta, snapshot=response)
                else:
                    h.on_message_delta(delta=delta, snapshot=response)

            # yield
            yield delta, response

        responses.append(response)

        # on message done
        if not is_tool_call:
            for h in handlers:
                h.on_message_done(response)
        new_messages.append(response.choices[0].message)

        # tool calls
        for tool_call in get_tool_calls(response):
            for h in handlers:
                h.on_tool_call_done(tool_call=tool_call)
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                h.on_tool_result(tool_message)
            new_messages.append(tool_message)

            yield None, tool_message

        if len(responses) >= (max_iterations or math.inf):
            break

    if response_callback:
        response_callback(Response(messages=new_messages, responses=responses))


async def completion_async(
    messages: list[Union[dict, Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations=None,
    handlers: list[Union[AsyncStreamHandler, StreamHandler]] = None,
    **kwargs,
) -> Response:
    """
    Perform asynchronous completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        call_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Returns:
        Response
    """
    response = None
    responses = []
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    while not response or has_tool_calls(response):
        response = await litellm.acompletion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools] if tools else None,
            **kwargs,
        )

        responses.append(response)

        # on message done
        for h in handlers:
            await maybe_coro(h.on_message_done(response))
        new_messages.append(response.choices[0].message)

        for tool_call in get_tool_calls(response):
            for h in handlers:
                await maybe_coro(h.on_tool_call_done(tool_call=tool_call))
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                await maybe_coro(h.on_tool_result(tool_message))
            new_messages.append(tool_message)

        if len(responses) >= (max_iterations or math.inf):
            break

    return Response(
        messages=new_messages,
        responses=responses,
    )


async def completion_stream_async(
    messages: list[Union[dict, Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations: int = None,
    handlers: list[Union[AsyncStreamHandler, StreamHandler]] = None,
    response_callback: Callable[[Response], None] = None,
    **kwargs,
) -> AsyncGenerator[Tuple[litellm.ModelResponse, litellm.ModelResponse], None]:
    """
    Perform asynchronous streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        call_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Yields:
        A tuple containing the current completion delta and the snapshot of the completion response.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """

    response = None
    responses = []
    new_messages = []

    if handlers is None:
        handlers = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    while not response or has_tool_calls(response):
        deltas = []
        is_tool_call = False
        async for delta in await litellm.acompletion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=True,
            **kwargs,
        ):
            deltas.append(delta)
            response = litellm.stream_chunk_builder(deltas)

            # on message / tool call created
            if len(deltas) == 1:
                if get_tool_calls(response):
                    is_tool_call = True
                for h in handlers:
                    if is_tool_call:
                        await maybe_coro(h.on_tool_call_created(delta=delta))
                    else:
                        await maybe_coro(h.on_message_created(delta=delta))

            # on message / tool call delta
            for h in handlers:
                if is_tool_call:
                    await maybe_coro(
                        h.on_tool_call_delta(delta=delta, snapshot=response)
                    )
                else:
                    await maybe_coro(h.on_message_delta(delta=delta, snapshot=response))

            # yield
            yield delta, response

        responses.append(response)

        # on message done
        if not is_tool_call:
            for h in handlers:
                await maybe_coro(h.on_message_done(response))
        new_messages.append(response.choices[0].message)

        # tool calls
        for tool_call in get_tool_calls(response):
            for h in handlers:
                await maybe_coro(h.on_tool_call_done(tool_call=tool_call))
            tool_message = handle_tool_call(tool_call, tools)
            for h in handlers:
                await maybe_coro(h.on_tool_result(tool_message))
            new_messages.append(tool_message)

            yield None, tool_message

        if len(responses) >= (max_iterations or math.inf):
            break

    if response_callback:
        response_callback(Response(messages=new_messages, responses=responses))
