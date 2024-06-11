import datetime
import math
from typing import AsyncGenerator, Callable, Generator, Optional, Union

import langchain_core.language_models as lc_models

import controlflow
import controlflow.llm.models
from controlflow.llm.handlers import (
    CompletionEvent,
    CompletionHandler,
    ResponseHandler,
)
from controlflow.llm.messages import AIMessage, AIMessageChunk, MessageType
from controlflow.llm.tools import (
    as_tools,
    handle_invalid_tool_call,
    handle_tool_call,
    handle_tool_call_async,
)


def _completion_generator(
    messages: list[MessageType],
    model: lc_models.BaseChatModel,
    tools: Optional[list[Callable]],
    max_iterations: int,
    stream: bool,
    ai_name: Optional[str],
    message_preprocessor: Callable = None,
    **kwargs,
) -> Generator[CompletionEvent, None, None]:
    response_messages = []
    response_message = None

    if tools:
        tools = as_tools(tools)
        model = model.bind_tools(tools)

    counter = 0
    try:
        yield CompletionEvent(type="start", payload={})

        # continue as long as the last response message contains tool calls (or
        # there is no response message yet)
        while not response_message or response_message.tool_calls:
            timestamp = datetime.datetime.now(datetime.timezone.utc)

            input_messages = messages + response_messages
            if message_preprocessor is not None:
                input_messages = message_preprocessor(input_messages)

            if not stream:
                response_message = model.invoke(
                    input=input_messages,
                    **kwargs,
                )
                response_message = AIMessage.from_message(
                    response_message, name=ai_name
                )

            else:
                deltas: list[AIMessageChunk] = []
                snapshot: AIMessageChunk = None

                for delta in model.stream(
                    input=input_messages,
                    **kwargs,
                ):
                    delta = AIMessageChunk.from_chunk(delta, name=ai_name)
                    deltas.append(delta)

                    if snapshot is None:
                        snapshot = delta
                    else:
                        snapshot = snapshot + delta

                    if len(deltas) == 1:
                        if delta.tool_call_chunks:
                            yield CompletionEvent(
                                type="tool_call_created", payload=dict(delta=delta)
                            )
                        else:
                            yield CompletionEvent(
                                type="message_created", payload=dict(delta=delta)
                            )

                    if delta.tool_call_chunks:
                        yield CompletionEvent(
                            type="tool_call_delta",
                            payload=dict(delta=delta, snapshot=snapshot),
                        )
                    else:
                        yield CompletionEvent(
                            type="message_delta",
                            payload=dict(delta=delta, snapshot=snapshot),
                        )

                # the last snapshot message is the response message
                response_message = snapshot.to_message()

            response_message.timestamp = timestamp

            if response_message.tool_calls or response_message.invalid_tool_calls:
                if response_message.tool_calls:
                    yield CompletionEvent(
                        type="tool_call_done", payload=dict(message=response_message)
                    )
                elif response_message.invalid_tool_calls:
                    yield CompletionEvent(
                        type="invalid_tool_call_done",
                        payload=dict(message=response_message),
                    )
            else:
                yield CompletionEvent(
                    type="message_done", payload=dict(message=response_message)
                )

            # append the response message to the list of response messages
            response_messages.append(response_message)

            # handle tool calls
            for tool_call in response_message.tool_calls:
                yield CompletionEvent(
                    type="tool_result_created",
                    payload=dict(message=response_message, tool_call=tool_call),
                )
                tool_result_message = handle_tool_call(tool_call, tools)
                response_messages.append(tool_result_message)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=tool_result_message)
                )

            # handle invalid tool calls
            for tool_call in response_message.invalid_tool_calls:
                invalid_tool_message = handle_invalid_tool_call(tool_call)
                response_messages.append(invalid_tool_message)

            counter += 1
            if counter >= (max_iterations or math.inf):
                break

    except (BaseException, Exception) as exc:
        yield CompletionEvent(type="exception", payload=dict(exc=exc))
        raise
    finally:
        yield CompletionEvent(type="end", payload={})


async def _completion_async_generator(
    messages: list[MessageType],
    model: lc_models.BaseChatModel,
    tools: Optional[list[Callable]],
    max_iterations: int,
    stream: bool,
    ai_name: Optional[str],
    message_preprocessor: Callable = None,
    **kwargs,
) -> AsyncGenerator[CompletionEvent, None]:
    response_messages = []
    response_message = None

    if tools:
        tools = as_tools(tools)
        model = model.bind_tools(tools)

    counter = 0
    try:
        yield CompletionEvent(type="start", payload={})

        # continue as long as the last response message contains tool calls (or
        # there is no response message yet)
        while not response_message or response_message.tool_calls:
            timestamp = datetime.datetime.now(datetime.timezone.utc)

            input_messages = messages + response_messages
            if message_preprocessor is not None:
                input_messages = message_preprocessor(input_messages)

            if not stream:
                response_message = await model.ainvoke(
                    input=input_messages,
                    tools=tools or None,
                    **kwargs,
                )
                response_message = AIMessage.from_message(
                    response_message, name=ai_name
                )

            else:
                deltas: list[AIMessageChunk] = []
                snapshot: AIMessageChunk = None

                async for delta in model.astream(
                    input=input_messages,
                    tools=tools or None,
                    **kwargs,
                ):
                    delta = AIMessageChunk.from_chunk(delta, name=ai_name)
                    deltas.append(delta)

                    if snapshot is None:
                        snapshot = delta
                    else:
                        snapshot = snapshot + delta

                    if len(deltas) == 1:
                        if delta.tool_call_chunks:
                            yield CompletionEvent(
                                type="tool_call_created", payload=dict(delta=delta)
                            )
                        else:
                            yield CompletionEvent(
                                type="message_created", payload=dict(delta=delta)
                            )

                    if delta.tool_call_chunks:
                        yield CompletionEvent(
                            type="tool_call_delta",
                            payload=dict(delta=delta, snapshot=snapshot),
                        )
                    else:
                        yield CompletionEvent(
                            type="message_delta",
                            payload=dict(delta=delta, snapshot=snapshot),
                        )

                # the last snapshot message is the response message
                response_message = snapshot.to_message()

            response_message.timestamp = timestamp

            if response_message.tool_calls or response_message.invalid_tool_calls:
                if response_message.tool_calls:
                    yield CompletionEvent(
                        type="tool_call_done", payload=dict(message=response_message)
                    )
                elif response_message.invalid_tool_calls:
                    yield CompletionEvent(
                        type="invalid_tool_call_done",
                        payload=dict(message=response_message),
                    )
            else:
                yield CompletionEvent(
                    type="message_done", payload=dict(message=response_message)
                )

            # append the response message to the list of response messages
            response_messages.append(response_message)

            # handle tool calls
            for tool_call in response_message.tool_calls:
                yield CompletionEvent(
                    type="tool_result_created",
                    payload=dict(message=response_message, tool_call=tool_call),
                )
                tool_result_message = await handle_tool_call_async(tool_call, tools)
                response_messages.append(tool_result_message)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=tool_result_message)
                )

            # handle invalid tool calls
            for tool_call in response_message.invalid_tool_calls:
                invalid_tool_message = handle_invalid_tool_call(tool_call)
                response_messages.append(invalid_tool_message)

            counter += 1
            if counter >= (max_iterations or math.inf):
                break

    except (BaseException, Exception) as exc:
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
    messages: list[MessageType],
    model: lc_models.BaseChatModel = None,
    tools: list[Callable] = None,
    max_iterations: int = None,
    handlers: list[CompletionHandler] = None,
    stream: bool = False,
    ai_name: Optional[str] = None,
    message_preprocessor: Callable = None,
    **kwargs,
) -> Union[list[MessageType], Generator[MessageType, None, None]]:
    if model is None:
        model = controlflow.llm.models.get_default_model()

    response_handler = ResponseHandler()
    handlers = handlers or []
    handlers.append(response_handler)

    completion_generator = _completion_generator(
        messages=messages,
        model=model,
        tools=tools,
        max_iterations=max_iterations,
        stream=stream,
        ai_name=ai_name,
        message_preprocessor=message_preprocessor,
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
    messages: list[MessageType],
    model: lc_models.BaseChatModel = None,
    tools: list[Callable] = None,
    max_iterations: int = None,
    handlers: list[CompletionHandler] = None,
    stream: bool = False,
    ai_name: Optional[str] = None,
    message_preprocessor: Callable = None,
    **kwargs,
) -> Union[list[MessageType], Generator[MessageType, None, None]]:
    if model is None:
        model = controlflow.llm.models.get_default_model()

    response_handler = ResponseHandler()
    handlers = handlers or []
    handlers.append(response_handler)

    completion_generator = _completion_async_generator(
        messages=messages,
        model=model,
        tools=tools,
        max_iterations=max_iterations,
        stream=stream,
        ai_name=ai_name,
        message_preprocessor=message_preprocessor,
        **kwargs,
    )

    handlers_generator = _handle_events_async(completion_generator, handlers)

    if stream:
        return handlers_generator
    else:
        async for _ in handlers_generator:
            pass
        return response_handler.response_messages
