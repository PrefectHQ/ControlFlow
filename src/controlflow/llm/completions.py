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
    ToolCall,
    as_tools,
    handle_invalid_tool_call,
    handle_tool_call,
    handle_tool_call_async,
)


def handle_delta_events(
    delta: AIMessageChunk, deltas: list[AIMessageChunk], snapshot: AIMessageChunk
):
    """
    Emit events for the given delta message.
    """
    if delta.content:
        if not deltas[-1].content:
            yield CompletionEvent(type="message_created", payload=dict(delta=delta))
        if delta.content != deltas[-1].content:
            yield CompletionEvent(
                type="message_delta",
                payload=dict(delta=delta, snapshot=snapshot),
            )

    if delta.tool_call_chunks:
        if not deltas[-1].tool_call_chunks:
            yield CompletionEvent(type="tool_call_created", payload=dict(delta=delta))
        yield CompletionEvent(
            type="tool_call_delta",
            payload=dict(delta=delta, snapshot=snapshot),
        )


def handle_done_events(message: AIMessage):
    """
    Emit events for the given message when it has been processed.
    """
    if message.content:
        yield CompletionEvent(type="message_done", payload=dict(message=message))
    if message.tool_calls:
        yield CompletionEvent(type="tool_call_done", payload=dict(message=message))
    if message.invalid_tool_calls:
        yield CompletionEvent(
            type="invalid_tool_call_done",
            payload=dict(message=message),
        )


def handle_multiple_talk_to_human_calls(tool_call: ToolCall, message: AIMessage):
    if (
        tool_call["name"] == "talk_to_human"
        and len([t for t in message.tool_calls if t["name"] == "talk_to_human"]) > 1
    ):
        error = 'Tool call "talk_to_human" can only be used once per turn.'
    else:
        error = None
    return error


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
            input_messages = messages + response_messages
            if message_preprocessor is not None:
                input_messages = message_preprocessor(input_messages)

            if not stream:
                response_message = model.invoke(input=input_messages, **kwargs)
                response_message = AIMessage.from_message(
                    response_message, name=ai_name
                )

            else:
                # initialize the list of deltas with an empty delta
                # to facilitate comparison with the previous delta
                deltas: list[AIMessageChunk] = [AIMessageChunk(content="")]
                snapshot: AIMessageChunk = None

                for delta in model.stream(input=input_messages, **kwargs):
                    delta = AIMessageChunk.from_chunk(delta, name=ai_name)

                    if snapshot is None:
                        snapshot = delta
                    else:
                        snapshot = snapshot + delta

                    yield from handle_delta_events(
                        delta=delta, deltas=deltas, snapshot=snapshot
                    )

                    deltas.append(delta)

                # the last snapshot message is the response message
                response_message = snapshot.to_message()

            # handle done events for the response message
            yield from handle_done_events(response_message)

            # append the response message to the list of response messages
            response_messages.append(response_message)

            # handle tool calls
            for tool_call in response_message.tool_calls:
                yield CompletionEvent(
                    type="tool_result_created",
                    payload=dict(message=response_message, tool_call=tool_call),
                )
                error = handle_multiple_talk_to_human_calls(tool_call, response_message)
                tool_result_message = handle_tool_call(tool_call, tools, error=error)
                response_messages.append(tool_result_message)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=tool_result_message)
                )

            # handle invalid tool calls
            for tool_call in response_message.invalid_tool_calls:
                yield CompletionEvent(
                    type="tool_result_created",
                    payload=dict(message=response_message, tool_call=tool_call),
                )
                invalid_tool_message = handle_invalid_tool_call(tool_call)
                response_messages.append(invalid_tool_message)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=invalid_tool_message)
                )

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
            input_messages = messages + response_messages
            if message_preprocessor is not None:
                input_messages = message_preprocessor(input_messages)

            if not stream:
                response_message = await model.ainvoke(input=input_messages, **kwargs)
                response_message = AIMessage.from_message(
                    response_message, name=ai_name
                )

            else:
                # initialize the list of deltas with an empty delta
                # to facilitate comparison with the previous delta
                deltas: list[AIMessageChunk] = [AIMessageChunk(content="")]
                snapshot: AIMessageChunk = None

                async for delta in model.astream(input=input_messages, **kwargs):
                    delta = AIMessageChunk.from_chunk(delta, name=ai_name)

                    if snapshot is None:
                        snapshot = delta
                    else:
                        snapshot = snapshot + delta

                    for event in handle_delta_events(
                        delta=delta, deltas=deltas, snapshot=snapshot
                    ):
                        yield event

                    deltas.append(delta)

                # the last snapshot message is the response message
                response_message = snapshot.to_message()

            # handle done events for the response message
            for event in handle_done_events(response_message):
                yield event

            # append the response message to the list of response messages
            response_messages.append(response_message)

            # handle tool calls
            for tool_call in response_message.tool_calls:
                yield CompletionEvent(
                    type="tool_result_created",
                    payload=dict(message=response_message, tool_call=tool_call),
                )
                error = handle_multiple_talk_to_human_calls(tool_call, response_message)
                tool_result_message = await handle_tool_call_async(
                    tool_call, tools, error=error
                )
                response_messages.append(tool_result_message)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=tool_result_message)
                )

            # handle invalid tool calls
            for tool_call in response_message.invalid_tool_calls:
                yield CompletionEvent(
                    type="tool_result_created",
                    payload=dict(message=response_message, tool_call=tool_call),
                )
                invalid_tool_message = handle_invalid_tool_call(tool_call)
                response_messages.append(invalid_tool_message)
                yield CompletionEvent(
                    type="tool_result_done", payload=dict(message=invalid_tool_message)
                )

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
            try:
                handler.on_event(event)
            except Exception as exc:
                generator.throw(exc)
        yield event


async def _handle_events_async(
    generator: AsyncGenerator, handlers: list[CompletionHandler]
) -> AsyncGenerator[CompletionEvent, None]:
    async for event in generator:
        for handler in handlers:
            try:
                handler.on_event(event)
            except Exception as exc:
                await generator.athrow(exc)
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
