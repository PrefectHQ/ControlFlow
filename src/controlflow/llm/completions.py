from typing import AsyncGenerator, Callable, Generator, Tuple, Union

import litellm

import controlflow
from controlflow.llm.tools import (
    function_to_tool_dict,
    handle_tool_calls,
    handle_tool_calls_async,
    has_tool_calls,
)
from controlflow.utilities.types import ControlFlowModel


class Response(ControlFlowModel):
    message: litellm.Message
    response: litellm.ModelResponse
    intermediate_messages: list[litellm.Message] = []
    intermediate_responses: list[litellm.ModelResponse] = []


def completion(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    use_tools=True,
    **kwargs,
) -> litellm.ModelResponse:
    """
    Perform completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        use_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Returns:
        A litellm.ModelResponse object representing the completion response.
    """

    intermediate_messages = []
    intermediate_responses = []

    if model is None:
        model = controlflow.settings.model

    tool_dicts = [function_to_tool_dict(tool) for tool in tools or []] or None

    response = litellm.completion(
        model=model,
        messages=messages,
        tools=tool_dicts,
        **kwargs,
    )

    while use_tools and has_tool_calls(response):
        intermediate_responses.append(response)
        intermediate_messages.append(response.choices[0].message)
        tool_messages = handle_tool_calls(response, tools)
        intermediate_messages.extend(tool_messages)
        response = litellm.completion(
            model=model,
            messages=messages + intermediate_messages,
            tools=tool_dicts,
            **kwargs,
        )

    return Response(
        message=response.choices[0].message,
        response=response,
        intermediate_messages=intermediate_messages,
        intermediate_responses=intermediate_responses,
    )


def stream_completion(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    use_tools: bool = True,
    **kwargs,
) -> Generator[Tuple[litellm.ModelResponse, litellm.ModelResponse], None, None]:
    """
    Perform streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        use_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Yields:
        A tuple containing the current completion chunk and the snapshot of the completion response.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """
    if model is None:
        model = controlflow.settings.model

    tool_dicts = [function_to_tool_dict(tool) for tool in tools or []] or None

    chunks = []
    for chunk in litellm.completion(
        model=model,
        messages=messages,
        stream=True,
        tools=tool_dicts,
        **kwargs,
    ):
        chunks.append(chunk)
        snapshot = litellm.stream_chunk_builder(chunks)
        yield chunk, snapshot

    response = snapshot

    while use_tools and has_tool_calls(response):
        messages.append(response.choices[0].message)
        tool_messages = handle_tool_calls(response, tools)
        messages.extend(tool_messages)
        chunks = []
        for chunk in litellm.completion(
            model=model,
            messages=messages,
            tools=tool_dicts,
            stream=True**kwargs,
        ):
            chunks.append(chunk)
            snapshot = litellm.stream_chunk_builder(chunks)
            yield chunk, snapshot
        response = snapshot


async def completion_async(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    use_tools=True,
    **kwargs,
) -> Response:
    """
    Perform asynchronous completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        use_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Returns:
        Response
    """
    intermediate_messages = []
    intermediate_responses = []

    if model is None:
        model = controlflow.settings.model

    tool_dicts = [function_to_tool_dict(tool) for tool in tools or []] or None

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tool_dicts,
        **kwargs,
    )

    while use_tools and has_tool_calls(response):
        intermediate_responses.append(response)
        intermediate_messages.append(response.choices[0].message)
        tool_messages = await handle_tool_calls_async(response, tools)
        intermediate_messages.extend(tool_messages)
        response = await litellm.acompletion(
            model=model,
            messages=messages + intermediate_messages,
            tools=tool_dicts,
            **kwargs,
        )

    return Response(
        message=response.choices[0].message,
        response=response,
        intermediate_messages=intermediate_messages,
        intermediate_responses=intermediate_responses,
    )


async def stream_completion_async(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    use_tools: bool = True,
    **kwargs,
) -> AsyncGenerator[Tuple[litellm.ModelResponse, litellm.ModelResponse], None]:
    """
    Perform asynchronous streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        use_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Yields:
        A tuple containing the current completion chunk and the snapshot of the completion response.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """
    if model is None:
        model = controlflow.settings.model

    tool_dicts = [function_to_tool_dict(tool) for tool in tools or []] or None

    chunks = []
    async for chunk in litellm.acompletion(
        model=model,
        messages=messages,
        stream=True,
        tools=tool_dicts,
        **kwargs,
    ):
        chunks.append(chunk)
        snapshot = litellm.stream_chunk_builder(chunks)
        yield chunk, snapshot

    response = snapshot

    while use_tools and has_tool_calls(response):
        messages.append(response.choices[0].message)
        tool_messages = await handle_tool_calls_async(response, tools)
        messages.extend(tool_messages)
        chunks = []
        async for chunk in litellm.acompletion(
            model=model,
            messages=messages,
            tools=tool_dicts,
            stream=True,
            **kwargs,
        ):
            chunks.append(chunk)
            snapshot = litellm.stream_chunk_builder(chunks)
            yield chunk, snapshot

        response = snapshot
