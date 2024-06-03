import math
from typing import AsyncGenerator, Callable, Generator, Union

import litellm
from litellm.utils import trim_messages

import controlflow
from controlflow.llm.handlers import (
    CompletionEvent,
    CompletionHandler,
    ResponseHandler,
)
from controlflow.llm.messages import (
    ControlFlowMessage,
    as_cf_messages,
    as_oai_messages,
)
from controlflow.llm.tools import (
    as_tools,
    get_tool_calls,
    handle_tool_call,
    handle_tool_call_async,
)


def _completion_generator(
    messages: list[Union[dict, ControlFlowMessage]],
    model: str,
    tools: list[Callable],
    assistant_name: str,
    max_iterations: int,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage],
    stream: bool,
    **kwargs,
) -> Generator[CompletionEvent, None, None]:
    response_messages = []
    response_message = None

    if "api_key" not in kwargs:
        kwargs["api_key"] = controlflow.settings.llm_api_key
    if "api_version" not in kwargs:
        kwargs["api_version"] = controlflow.settings.llm_api_version
    if "api_base" not in kwargs:
        kwargs["api_base"] = controlflow.settings.llm_api_base

    tools = as_tools(tools or [])

    counter = 0
    try:
        yield CompletionEvent(type="start", payload={})

        # continue as long as the last response message contains tool calls (or
        # there is no response message yet)
        while not response_message or get_tool_calls(response_message):
            # the input messages are the provided messages plus all response messages
            # including tool calls and results
            input_messages = as_oai_messages(messages + response_messages)
            # apply message preprocessor if provided
            if message_preprocessor:
                input_messages = [
                    m
                    for msg in input_messages
                    if (m := message_preprocessor(msg)) is not None
                ]
            response = litellm.completion(
                model=model,
                messages=trim_messages(input_messages, model=model),
                tools=[t.model_dump() for t in tools] if tools else None,
                stream=stream,
                **kwargs,
            )

            # if streaming is enabled, we need to handle the deltas
            if stream:
                deltas = []
                for delta in response:
                    deltas.append(delta)
                    snapshot = litellm.stream_chunk_builder(deltas)
                    delta_message, snapshot_message = as_cf_messages([delta, snapshot])
                    delta_message.name, snapshot_message.name = (
                        assistant_name,
                        assistant_name,
                    )

                    if len(deltas) == 1:
                        yield CompletionEvent(
                            type="message_created", payload=dict(delta=delta_message)
                        )

                    yield CompletionEvent(
                        type="message_delta",
                        payload=dict(delta=delta_message, snapshot=snapshot_message),
                    )
                # the last snapshot message is the response message
                response_message = snapshot_message

            else:
                [response_message] = as_cf_messages([response])
                response_message.name = assistant_name

            if response_message.has_tool_calls():
                yield CompletionEvent(
                    type="tool_call_done", payload=dict(message=response_message)
                )
            else:
                yield CompletionEvent(
                    type="message_done", payload=dict(message=response_message)
                )

            # append the response message to the list of response messages
            response_messages.append(response_message)

            # handle tool calls
            for tool_call in get_tool_calls(response_message):
                tool_result_message = handle_tool_call(tool_call, tools)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=tool_result_message)
                )
                response_messages.append(tool_result_message)

            counter += 1
            if counter >= (max_iterations or math.inf):
                break

    except Exception as exc:
        yield CompletionEvent(type="exception", payload=dict(exc=exc))
        raise
    finally:
        yield CompletionEvent(type="end", payload={})


async def _completion_async_generator(
    messages: list[Union[dict, ControlFlowMessage]],
    model: str,
    tools: list[Callable],
    assistant_name: str,
    max_iterations: int,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage],
    stream: bool,
    **kwargs,
) -> AsyncGenerator[CompletionEvent, None]:
    response_messages = []
    response_message = None

    if "api_key" not in kwargs:
        kwargs["api_key"] = controlflow.settings.llm_api_key
    if "api_version" not in kwargs:
        kwargs["api_version"] = controlflow.settings.llm_api_version
    if "api_base" not in kwargs:
        kwargs["api_base"] = controlflow.settings.llm_api_base

    tools = as_tools(tools or [])

    counter = 0
    try:
        yield CompletionEvent(type="start", payload={})

        while not response_message or get_tool_calls(response_message):
            input_messages = as_oai_messages(messages + response_messages)
            if message_preprocessor:
                input_messages = [
                    m
                    for msg in input_messages
                    if (m := message_preprocessor(msg)) is not None
                ]

            response = await litellm.acompletion(
                model=model,
                messages=trim_messages(input_messages, model=model),
                tools=[t.model_dump() for t in tools] if tools else None,
                stream=stream,
                **kwargs,
            )

            if stream:
                deltas = []
                async for delta in response:
                    deltas.append(delta)
                    snapshot = litellm.stream_chunk_builder(deltas)
                    delta_message, snapshot_message = as_cf_messages([delta, snapshot])
                    delta_message.name, snapshot_message.name = (
                        assistant_name,
                        assistant_name,
                    )

                    if len(deltas) == 1:
                        yield CompletionEvent(
                            type="message_created", payload=dict(delta=delta_message)
                        )

                    yield CompletionEvent(
                        type="message_delta",
                        payload=dict(delta=delta_message, snapshot=snapshot_message),
                    )
                response_message = snapshot_message

            else:
                [response_message] = as_cf_messages([response])
                response_message.name = assistant_name

            if response_message.has_tool_calls():
                yield CompletionEvent(
                    type="tool_call_done", payload=dict(message=response_message)
                )
            else:
                yield CompletionEvent(
                    type="message_done", payload=dict(message=response_message)
                )

            response_messages.append(response_message)

            for tool_call in get_tool_calls(response_message):
                tool_result_message = await handle_tool_call_async(tool_call, tools)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=tool_result_message)
                )
                response_messages.append(tool_result_message)

            counter += 1
            if counter >= (max_iterations or math.inf):
                break

    except Exception as exc:
        yield CompletionEvent(type="exception", payload=dict(exc=exc))
        raise
    finally:
        yield CompletionEvent(type="end", payload={})


def _handle_events(
    generator: Generator[CompletionEvent, None, None], handlers: list[CompletionHandler]
) -> Generator[CompletionEvent, None, None]:
    for event in generator:
        for handler in handlers:
            handler.on_event(event)
        yield event


async def _handle_events_async(
    generator: AsyncGenerator, handlers: list[CompletionHandler]
) -> AsyncGenerator[CompletionEvent, None]:
    async for event in generator:
        for handler in handlers:
            handler.on_event(event)
        yield event


def completion(
    messages: list[Union[dict, ControlFlowMessage]],
    model: str = None,
    tools: list[Callable] = None,
    assistant_name: str = None,
    max_iterations: int = None,
    handlers: list[CompletionHandler] = None,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage] = None,
    stream: bool = False,
    **kwargs,
) -> Union[list[ControlFlowMessage], Generator[ControlFlowMessage, None, None]]:
    if model is None:
        model = controlflow.settings.llm_model

    response_handler = ResponseHandler()
    handlers = handlers or []
    handlers.append(response_handler)

    completion_generator = _completion_generator(
        messages=messages,
        model=model,
        tools=tools,
        assistant_name=assistant_name,
        max_iterations=max_iterations,
        message_preprocessor=message_preprocessor,
        stream=stream,
        **kwargs,
    )

    handlers_generator = _handle_events(completion_generator, handlers)

    if stream:
        return handlers_generator
    else:
        for _ in handlers_generator:
            pass
        return response_handler.response_messages


async def completion_async(
    messages: list[Union[dict, ControlFlowMessage]],
    model: str = None,
    tools: list[Callable] = None,
    assistant_name: str = None,
    max_iterations: int = None,
    handlers: list[CompletionHandler] = None,
    message_preprocessor: Callable[[ControlFlowMessage], ControlFlowMessage] = None,
    stream: bool = False,
    **kwargs,
) -> Union[list[ControlFlowMessage], Generator[ControlFlowMessage, None, None]]:
    if model is None:
        model = controlflow.settings.llm_model

    response_handler = ResponseHandler()
    handlers = handlers or []
    handlers.append(response_handler)

    completion_generator = _completion_async_generator(
        messages=messages,
        model=model,
        tools=tools,
        assistant_name=assistant_name,
        max_iterations=max_iterations,
        message_preprocessor=message_preprocessor,
        stream=stream,
        **kwargs,
    )

    handlers_generator = _handle_events_async(completion_generator, handlers)

    if stream:
        return handlers_generator
    else:
        async for _ in handlers_generator:
            pass
        return response_handler.response_messages
