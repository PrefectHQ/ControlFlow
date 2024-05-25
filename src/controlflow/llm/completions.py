import math
from typing import AsyncGenerator, Callable, Generator, Tuple, Union

import litellm
from litellm.utils import trim_messages

import controlflow
from controlflow.llm.handlers import CompletionHandler, CompoundHandler
from controlflow.llm.messages import (
    ControlFlowMessage,
    as_cf_messages,
    as_oai_messages,
)
from controlflow.llm.tools import (
    as_tools,
    get_tool_calls,
    handle_tool_call,
)


def completion(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    assistant_name: str = None,
    max_iterations=None,
    handlers: list[CompletionHandler] = None,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage] = None,
    stream: bool = False,
    **kwargs,
) -> list[ControlFlowMessage]:
    """
    Perform completion using the LLM model.

    Args:
        messages (list[Union[dict, ControlFlowMessage]]): A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools (list[Callable]): A list of callable tools to be used during completion.
        assistant_name (str): The name of the assistant, which will be set as the `name` attribute of any messages it generates.
        max_iterations: The maximum number of iterations to perform completion. If not provided, it will continue until there are no more response messages or tool calls.
        handlers (list[CompletionHandler]): A list of completion handlers to be used during completion.
        message_preprocessor (Callable[[ControlFlowMessage], ControlFlowMessage]): A callable function to preprocess the completion messages before sending them to the LLM model.
        stream (bool): If True, stream the completion response. Deltas will be passed to the handler as they are received; complete messages will be yielded as well.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Returns:
        list[ControlFlowMessage]: A list of ControlFlowMessage objects representing the completion response.
    """
    if stream:
        return _completion_stream(
            messages=messages,
            model=model,
            tools=tools,
            assistant_name=assistant_name,
            max_iterations=max_iterations,
            handlers=handlers,
            message_preprocessor=message_preprocessor,
            **kwargs,
        )

    response_messages = []
    new_messages = []

    handler = CompoundHandler(handlers=handlers or [])

    if model is None:
        model = controlflow.settings.llm_model
    if "api_key" not in kwargs:
        kwargs["api_key"] = controlflow.settings.llm_api_key
    if "api_version" not in kwargs:
        kwargs["api_version"] = controlflow.settings.llm_api_version
    if "api_base" not in kwargs:
        kwargs["api_base"] = controlflow.settings.llm_api_base

    tools = as_tools(tools or [])

    handler.on_start()

    counter = 0
    try:
        while not response_messages or get_tool_calls(response_messages):
            completion_messages = as_oai_messages(messages + new_messages)
            if message_preprocessor:
                completion_messages = [
                    m
                    for msg in completion_messages
                    if (m := message_preprocessor(msg)) is not None
                ]
            response = litellm.completion(
                model=model,
                messages=trim_messages(completion_messages, model=model),
                tools=[t.model_dump() for t in tools] if tools else None,
                **kwargs,
            )

            response_messages = as_cf_messages([response])

            # on message done
            for msg in response_messages:
                msg.name = assistant_name
                new_messages.append(msg)
                if msg.has_tool_calls():
                    handler.on_tool_call_done(msg)
                else:
                    handler.on_message_done(msg)

            # tool calls
            for tool_call in get_tool_calls(response_messages):
                tool_message = handle_tool_call(tool_call, tools)
                handler.on_tool_result(tool_message)
                new_messages.append(tool_message)

            counter += 1
            if counter >= (max_iterations or math.inf):
                break

        return new_messages
    except Exception as exc:
        handler.on_exception(exc)
        raise
    finally:
        handler.on_end()


def _completion_stream(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    assistant_name: str = None,
    max_iterations: int = None,
    handlers: list[CompletionHandler] = None,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage] = None,
    **kwargs,
) -> Generator[Tuple[litellm.ModelResponse, litellm.ModelResponse], None, None]:
    """
    Perform streaming completion using the LLM model.

    Args:
        messages (list[Union[dict, ControlFlowMessage]]): A list of messages to be used for completion.
        model (optional): The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools (optional): A list of callable tools to be used during completion.
        assistant_name (optional): The name of the assistant, which will be set as the `name` attribute of any messages it generates.
        max_iterations (optional): The maximum number of iterations to perform. If not provided, it will iterate indefinitely.
        handlers (optional): A list of completion handlers to be used during completion.
        message_preprocessor (optional): A callable function to preprocess each message before completion.
        **kwargs: Additional keyword arguments to be passed to the litellm.completion function.

    Yields:
        Each message generated during completion.

    Returns:
        The final completion response as a litellm.ModelResponse object.
    """

    snapshot_message = None
    new_messages = []

    handler = CompoundHandler(handlers=handlers or [])
    if model is None:
        model = controlflow.settings.llm_model
    if "api_key" not in kwargs:
        kwargs["api_key"] = controlflow.settings.llm_api_key
    if "api_version" not in kwargs:
        kwargs["api_version"] = controlflow.settings.llm_api_version
    if "api_base" not in kwargs:
        kwargs["api_base"] = controlflow.settings.llm_api_base

    tools = as_tools(tools or [])

    handler.on_start()

    counter = 0
    try:
        while not snapshot_message or get_tool_calls(snapshot_message):
            completion_messages = as_oai_messages(messages + new_messages)
            if message_preprocessor:
                completion_messages = [
                    m
                    for msg in completion_messages
                    if (m := message_preprocessor(msg)) is not None
                ]

            response = litellm.completion(
                model=model,
                messages=trim_messages(completion_messages, model=model),
                tools=[t.model_dump() for t in tools] if tools else None,
                stream=True,
                **kwargs,
            )

            deltas = []
            for delta in response:
                deltas.append(delta)
                snapshot = litellm.stream_chunk_builder(deltas)
                delta_message, snapshot_message = as_cf_messages([delta, snapshot])
                delta_message.name, snapshot_message.name = (
                    assistant_name,
                    assistant_name,
                )

                # on message created
                if len(deltas) == 1:
                    if snapshot_message.has_tool_calls():
                        handler.on_tool_call_created(delta=delta_message)
                    else:
                        handler.on_message_created(delta=delta_message)

                # on message delta
                if snapshot_message.has_tool_calls():
                    handler.on_tool_call_delta(
                        delta=delta_message, snapshot=snapshot_message
                    )
                else:
                    handler.on_message_delta(
                        delta=delta_message, snapshot=snapshot_message
                    )

            new_messages.append(snapshot_message)
            yield snapshot_message

            # on message done
            if snapshot_message.has_tool_calls():
                handler.on_tool_call_done(snapshot_message)
            else:
                handler.on_message_done(snapshot_message)

            # tool calls
            for tool_call in get_tool_calls(snapshot_message):
                tool_message = handle_tool_call(tool_call, tools)
                handler.on_tool_result(tool_message)
                new_messages.append(tool_message)
                yield tool_message

            counter += 1
            if counter >= (max_iterations or math.inf):
                break
    except Exception as exc:
        handler.on_exception(exc)
        raise
    finally:
        handler.on_end()


async def completion_async(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    assistant_name: str = None,
    max_iterations=None,
    handlers: list[CompletionHandler] = None,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage] = None,
    stream: bool = False,
    **kwargs,
) -> list[ControlFlowMessage]:
    """
    Perform asynchronous completion using the LLM model.

    Args:
        messages (list[Union[dict, ControlFlowMessage]]): A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools (list[Callable]): A list of callable tools to be used during completion.
        assistant_name (str): The name of the assistant, which will be set as the `name` attribute of any messages it generates.
        max_iterations: The maximum number of iterations to perform. If not provided, it will iterate until completion is done.
        handlers (list[CompletionHandler]): A list of completion handlers to be used during completion.
        message_preprocessor (Callable[[ControlFlowMessage], ControlFlowMessage]): A callable function to preprocess the completion messages.
        stream (bool): If True, stream the completion response. Deltas will be passed to the handler as they are received; complete messages will be yielded as well.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Returns:
        list[ControlFlowMessage]: A list of ControlFlowMessage objects representing the completion response.
    """
    if stream:
        return _completion_stream_async(
            messages=messages,
            model=model,
            tools=tools,
            assistant_name=assistant_name,
            max_iterations=max_iterations,
            handlers=handlers,
            message_preprocessor=message_preprocessor,
            **kwargs,
        )
    response_messages = []
    new_messages = []

    handler = CompoundHandler(handlers=handlers or [])
    if model is None:
        model = controlflow.settings.llm_model
    if "api_key" not in kwargs:
        kwargs["api_key"] = controlflow.settings.llm_api_key
    if "api_version" not in kwargs:
        kwargs["api_version"] = controlflow.settings.llm_api_version
    if "api_base" not in kwargs:
        kwargs["api_base"] = controlflow.settings.llm_api_base

    tools = as_tools(tools or [])

    handler.on_start()

    try:
        counter = 0
        while not response_messages or get_tool_calls(response_messages):
            completion_messages = as_oai_messages(messages + new_messages)
            if message_preprocessor:
                completion_messages = [
                    m
                    for msg in completion_messages
                    if (m := message_preprocessor(msg)) is not None
                ]

            response = await litellm.acompletion(
                model=model,
                messages=trim_messages(completion_messages, model=model),
                tools=[t.model_dump() for t in tools] if tools else None,
                **kwargs,
            )

            response_messages = as_cf_messages([response])

            # on done
            for msg in response_messages:
                msg.name = assistant_name
                new_messages.append(msg)
                if msg.has_tool_calls():
                    handler.on_tool_call_done(msg)
                else:
                    handler.on_message_done(msg)

            # tool calls
            for tool_call in get_tool_calls(response_messages):
                tool_message = handle_tool_call(tool_call, tools)
                handler.on_tool_result(tool_message)
                new_messages.append(tool_message)

            counter += 1
            if counter >= (max_iterations or math.inf):
                break

        return new_messages
    except Exception as exc:
        handler.on_exception(exc)
        raise
    finally:
        handler.on_end()


"""
Perform asynchronous streaming completion using the LLM model.

Args:
    messages: A list of messages to be used for completion.
    model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
    tools: A list of callable tools to be used during completion.
    assistant_name: The name of the assistant, which will be set as the `name` attribute of any messages it generates.
    max_iterations: The maximum number of iterations to perform completion. If not provided, it will continue until completion is done.
    handlers: A list of CompletionHandler objects to handle completion events.
    message_preprocessor: A callable function to preprocess each ControlFlowMessage before completion.
    **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

Yields:
    Each ControlFlowMessage generated during completion.

Returns:
    The final completion response as a list of ControlFlowMessage objects.
"""


async def _completion_stream_async(
    messages: list[Union[dict, ControlFlowMessage]],
    model=None,
    tools: list[Callable] = None,
    assistant_name: str = None,
    max_iterations: int = None,
    handlers: list[CompletionHandler] = None,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage] = None,
    **kwargs,
) -> AsyncGenerator[ControlFlowMessage, None]:
    """
    Perform asynchronous streaming completion using the LLM model.

    Args:
        messages: A list of messages to be used for completion.
        model: The LLM model to be used for completion. If not provided, the default model from controlflow.settings will be used.
        tools: A list of callable tools to be used during completion.
        assistant_name: The name of the assistant, which will be set as the `name` attribute of any messages it generates.
        **kwargs: Additional keyword arguments to be passed to the litellm.acompletion function.

    Yields:
        Each message

    Returns:
        The final completion response as a list of ControlFlowMessage objects.
    """

    snapshot_message = None
    new_messages = []

    handler = CompoundHandler(handlers=handlers or [])
    if model is None:
        model = controlflow.settings.llm_model
    if "api_key" not in kwargs:
        kwargs["api_key"] = controlflow.settings.llm_api_key
    if "api_version" not in kwargs:
        kwargs["api_version"] = controlflow.settings.llm_api_version
    if "api_base" not in kwargs:
        kwargs["api_base"] = controlflow.settings.llm_api_base

    tools = as_tools(tools or [])

    handler.on_start()

    counter = 0
    try:
        while not snapshot_message or get_tool_calls(snapshot_message):
            completion_messages = as_oai_messages(messages + new_messages)
            if message_preprocessor:
                completion_messages = [
                    m
                    for msg in completion_messages
                    if (m := message_preprocessor(msg)) is not None
                ]

            response = await litellm.acompletion(
                model=model,
                messages=trim_messages(completion_messages, model=model),
                tools=[t.model_dump() for t in tools] if tools else None,
                stream=True,
                **kwargs,
            )

            deltas = []
            async for delta in response:
                deltas.append(delta)
                snapshot = litellm.stream_chunk_builder(deltas)
                delta_message, snapshot_message = as_cf_messages([delta, snapshot])
                delta_message.name, snapshot_message.name = (
                    assistant_name,
                    assistant_name,
                )

                # on message created
                if len(deltas) == 1:
                    if snapshot_message.has_tool_calls():
                        handler.on_tool_call_created(delta=delta_message)
                    else:
                        handler.on_message_created(delta=delta_message)

                # on message delta
                if snapshot_message.has_tool_calls():
                    handler.on_tool_call_delta(
                        delta=delta_message, snapshot=snapshot_message
                    )
                else:
                    handler.on_message_delta(
                        delta=delta_message, snapshot=snapshot_message
                    )

            # on message done
            if snapshot_message.has_tool_calls():
                handler.on_tool_call_done(snapshot_message)
            else:
                handler.on_message_done(snapshot_message)

            new_messages.append(snapshot_message)
            yield snapshot_message

            # tool calls
            for tool_call in get_tool_calls(snapshot_message):
                tool_message = handle_tool_call(tool_call, tools)
                handler.on_tool_result(tool_message)
                new_messages.append(tool_message)
                yield tool_message

            counter += 1
            if counter >= (max_iterations or math.inf):
                break
    except Exception as exc:
        handler.on_exception(exc)
    finally:
        handler.on_end()
