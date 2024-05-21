import inspect
from typing import Generator, Optional

import litellm

from controlflow.llm.tools import ToolCall
from controlflow.utilities.types import Message


class StreamHandler:
    def stream(
        self,
        gen: Generator[
            tuple[Optional[litellm.ModelResponse], litellm.ModelResponse], None, None
        ],
    ):
        last_snapshot = None
        for delta, snapshot in gen:
            snapshot_message = snapshot.choices[0].message

            # handle tool call outputs
            if delta is None and snapshot_message.role == "tool":
                self.on_tool_call(tool_call=snapshot_message._tool_call)
                self.on_message_done(snapshot_message)
                continue

            delta_message = delta.choices[0].delta

            # handle new messages
            if not last_snapshot or snapshot.id != last_snapshot.id:
                self.on_message_created(delta_message)

            # handle updated messages
            self.on_message_delta(delta=delta_message, snapshot=snapshot_message)

            # handle completed messages
            if delta.choices[0].finish_reason:
                self.on_message_done(snapshot_message)

            last_snapshot = snapshot

    def on_message_created(self, delta: litellm.utils.Delta):
        pass

    def on_message_delta(self, delta: litellm.utils.Delta, snapshot: litellm.Message):
        pass

    def on_message_done(self, message: Message):
        pass

    def on_tool_call(self, tool_call: ToolCall):
        pass


async def _maybe_coro(maybe_coro):
    if inspect.isawaitable(maybe_coro):
        return await maybe_coro


class AsyncStreamHandler(StreamHandler):
    async def stream(
        self,
        gen: Generator[
            tuple[Optional[litellm.ModelResponse], litellm.ModelResponse], None, None
        ],
    ):
        last_snapshot = None
        async for delta, snapshot in gen:
            snapshot_message = snapshot.choices[0].message

            # handle tool call outputs
            if delta is None and snapshot_message.role == "tool":
                await _maybe_coro(
                    self.on_tool_call(tool_call=snapshot_message._tool_call)
                )
                await _maybe_coro(self.on_message_done(snapshot_message))
                continue

            delta_message = delta.choices[0].delta

            # handle new messages
            if not last_snapshot or snapshot.id != last_snapshot.id:
                await _maybe_coro(self.on_message_created(delta_message))

            # handle updated messages
            await _maybe_coro(
                self.on_message_delta(delta=delta_message, snapshot=snapshot_message)
            )

            # handle completed messages
            if delta.choices[0].finish_reason:
                await _maybe_coro(self.on_message_done(snapshot_message))

            last_snapshot = snapshot

    async def on_message_created(self, delta: litellm.utils.Delta):
        pass

    async def on_message_delta(
        self, delta: litellm.utils.Delta, snapshot: litellm.Message
    ):
        pass

    async def on_message_done(self, message: Message):
        pass

    async def on_tool_call(self, tool_call: ToolCall):
        pass
