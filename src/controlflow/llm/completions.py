from typing import AsyncGenerator, Callable, Generator, Tuple, Union

import litellm
from litellm.utils import trim_messages
from pydantic import computed_field

import controlflow
from controlflow.llm.tools import (
    as_tools,
    handle_tool_calls,
    handle_tool_calls_async,
    has_tool_calls,
)
from controlflow.utilities.types import ControlFlowModel


class Response(ControlFlowModel):
    messages: list[litellm.Message] = []
    responses: list[litellm.ModelResponse] = []

    @computed_field
    def last_message(self) -> litellm.Message:
        return self.messages[-1] if self.messages else None

    @computed_field
    def last_response(self) -> litellm.ModelResponse:
        return self.responses[-1] if self.responses else None


def completion(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    call_tools=True,
    **kwargs,
) -> litellm.ModelResponse:
    """
    Perform completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        call_tools: A boolean indicating whether to use the provided tools during completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Returns:
        A litellm.ModelResponse object representing the completion response.
    """

    new_messages = []
    new_responses = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or []) or None

    response = litellm.completion(
        model=model,
        messages=trim_messages(messages, model=model),
        tools=[t.model_dump() for t in tools],
        **kwargs,
    )

    new_responses.append(response)
    new_messages.append(response.choices[0].message)

    while call_tools and has_tool_calls(response):
        new_messages.extend(handle_tool_calls(response, tools))

        response = litellm.completion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools],
            **kwargs,
        )

        new_responses.append(response)
        new_messages.append(response.choices[0].message)

    return Response(
        messages=new_messages,
        responses=new_responses,
    )


def stream_completion(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    call_tools: bool = True,
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
        A tuple containing the current completion chunk and the snapshot of the completion response.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """
    new_messages = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or []) or None

    chunks = []
    for chunk in litellm.completion(
        model=model,
        messages=trim_messages(messages, model=model),
        stream=True,
        tools=[t.model_dump() for t in tools],
        **kwargs,
    ):
        chunks.append(chunk)
        response = litellm.stream_chunk_builder(chunks)
        yield chunk, response

    new_messages.append(response.choices[0].message)

    while call_tools and has_tool_calls(response):
        new_messages.extend(handle_tool_calls(response, tools))
        chunks = []

        for chunk in litellm.completion(
            model=model,
            messages=trim_messages(messages, model=model),
            tools=[t.model_dump() for t in tools],
            stream=True**kwargs,
        ):
            chunks.append(chunk)
            response = litellm.stream_chunk_builder(chunks)
            yield chunk, response

        new_messages.append(response.choices[0].message)


async def completion_async(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    call_tools=True,
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
    new_messages = []
    new_responses = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or []) or None

    response = await litellm.acompletion(
        model=model,
        messages=trim_messages(messages, model=model),
        tools=[t.model_dump() for t in tools],
        **kwargs,
    )

    new_responses.append(response)
    new_messages.append(response.choices[0].message)

    while call_tools and has_tool_calls(response):
        new_messages.extend(await handle_tool_calls_async(response, tools))

        response = await litellm.acompletion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools],
            **kwargs,
        )

        new_responses.append(response)
        new_messages.append(response.choices[0].message)

    return Response(
        messages=new_messages,
        responses=new_responses,
    )


async def stream_completion_async(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    call_tools: bool = True,
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
        A tuple containing the current completion chunk and the snapshot of the completion response.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """
    new_messages = []

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or []) or None

    chunks = []
    async for chunk in litellm.acompletion(
        model=model,
        messages=trim_messages(messages, model=model),
        stream=True,
        tools=[t.model_dump() for t in tools],
        **kwargs,
    ):
        chunks.append(chunk)
        response = litellm.stream_chunk_builder(chunks)
        yield chunk, response

    new_messages.append(response.choices[0].message)

    while call_tools and has_tool_calls(response):
        new_messages.extend(await handle_tool_calls_async(response, tools))
        chunks = []

        async for chunk in litellm.acompletion(
            model=model,
            messages=trim_messages(messages + new_messages, model=model),
            tools=[t.model_dump() for t in tools],
            stream=True,
            **kwargs,
        ):
            chunks.append(chunk)
            response = litellm.stream_chunk_builder(chunks)
            yield chunk, response

        new_messages.append(response.choices[0].message)
