import datetime
from typing import Union

from textual.reactive import reactive
from textual.widgets import Static

from controlflow.llm.formatting import format_message, format_tool_message
from controlflow.llm.messages import AIMessage, ToolMessage, UserMessage


def format_timestamp(timestamp: datetime.datetime) -> str:
    return timestamp.strftime("%l:%M:%S %p")


class TUIMessage(Static):
    message: reactive[Union[UserMessage, AIMessage]] = reactive(
        None, always_update=True, layout=True
    )

    def __init__(self, message: Union[UserMessage, AIMessage], **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def render(self):
        return format_message(self.message)


class TUIToolMessage(Static):
    message: reactive[ToolMessage] = reactive(None, always_update=True, layout=True)

    def __init__(self, message: ToolMessage, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def render(self):
        return format_tool_message(self.message)
