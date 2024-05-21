import math
from typing import AsyncGenerator, Callable, Generator, Optional, Tuple, Union

import litellm
from litellm.utils import trim_messages

import controlflow
from controlflow.llm.tools import (
    as_tools,
    handle_tool_calls,
    handle_tool_calls_async,
    handle_tool_calls_gen,
    handle_tool_calls_gen_async,
    has_tool_calls,
)
from controlflow.utilities.types import ControlFlowModel, ToolCall


class Response(ControlFlowModel):
    messages: list[litellm.Message] = []
    responses: list[litellm.ModelResponse] = []

    def last_message(self) -> Optional[litellm.Message]:
        return self.messages[-1] if self.messages else None

    def last_response(self) -> Optional[litellm.ModelResponse]:
        return self.responses[-1] if self.responses else None

    def tool_calls(self) -> list[ToolCall]:
        return [
            m["_tool_call"]
            for m in self.messages
            if m.role == "tool" and m.get("_tool_call") is not None
        ]


def completion(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations=None,
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
        new_messages.append(response.choices[0].message)
        new_messages.extend(handle_tool_calls(response, tools))

        if len(responses) >= (max_iterations or math.inf):
            break

    return Response(
        messages=new_messages,
        responses=responses,
    )


def completion_stream(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations: int = None,
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
    messages = messages.copy()

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    i = 0
    while not response or has_tool_calls(response):
        deltas = []

        for delta in litellm.completion(
            model=model,
            messages=trim_messages(messages, model=model),
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=True,
            **kwargs,
        ):
            deltas.append(delta)
            response = litellm.stream_chunk_builder(deltas)
            yield delta, response

        for tool_msg in handle_tool_calls_gen(response, tools):
            messages.append(tool_msg)
            yield None, tool_msg

        messages.append(response.choices[0].message)

        i += 1
        if i >= (max_iterations or math.inf):
            break


async def completion_async(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations=None,
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
        new_messages.append(response.choices[0].message)
        new_messages.extend(await handle_tool_calls_async(response, tools))

    return Response(
        messages=new_messages,
        responses=responses,
    )


async def completion_stream_async(
    messages: list[Union[dict, litellm.Message]],
    model=None,
    tools: list[Callable] = None,
    max_iterations: int = None,
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
    messages = messages.copy()

    if model is None:
        model = controlflow.settings.model

    tools = as_tools(tools or [])

    i = 0
    while not response or has_tool_calls(response):
        deltas = []

        async for delta in litellm.acompletion(
            model=model,
            messages=trim_messages(messages, model=model),
            tools=[t.model_dump() for t in tools] if tools else None,
            stream=True,
            **kwargs,
        ):
            deltas.append(delta)
            response = litellm.stream_chunk_builder(deltas)
            yield delta, response

        async for tool_msg in handle_tool_calls_gen_async(response, tools):
            messages.append(tool_msg)
            yield None, tool_msg
        messages.append(response.choices[0].message)

        i += 1
        if i >= (max_iterations or math.inf):
            break
