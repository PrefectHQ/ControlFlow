from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import tiktoken

import controlflow
from controlflow.events.events import Event
from controlflow.llm.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from controlflow.llm.rules import LLMRules
from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.agents.agent import Agent
    from controlflow.flows.flow import Flow
    from controlflow.orchestration.controller import Controller
    from controlflow.tasks.task import Task
logger = get_logger(__name__)


def add_user_message_to_beginning(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    """
    If the LLM requires the user message to be the first message, add a user
    message to the beginning of the list.
    """
    if rules.require_user_message_after_system:
        if not messages or not isinstance(messages[0], HumanMessage):
            messages.insert(0, HumanMessage(content="SYSTEM: Begin."))
    return messages


def ensure_at_least_one_message(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    if not messages and rules.require_at_least_one_message:
        messages.append(HumanMessage(content="SYSTEM: Begin."))
    return messages


def add_user_message_to_end(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    """
    If the LLM doesn't allow the last message to be from the AI when using tools,
    add a user message to the end of the list.
    """
    if not rules.allow_last_message_from_ai_when_using_tools:
        if not messages or isinstance(messages[-1], AIMessage):
            msg = HumanMessage(content="SYSTEM: Continue.")
            messages.append(msg)
    return messages


def remove_duplicate_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Removes duplicate messages from the list.
    """
    seen = set()
    new_messages = []
    for message in messages:
        if message.id not in seen:
            new_messages.append(message)
            if message.id:
                seen.add(message.id)
    return new_messages


def break_up_consecutive_ai_messages(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    """
    Breaks up consecutive AI messages by inserting a system message.
    """
    if not messages or rules.allow_consecutive_ai_messages:
        return messages

    new_messages = messages.copy()
    i = 1
    while i < len(new_messages):
        if isinstance(new_messages[i], AIMessage) and isinstance(
            new_messages[i - 1], AIMessage
        ):
            new_messages.insert(i, SystemMessage(content="Continue."))
        i += 1

    return new_messages


def convert_system_messages(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    """
    Converts system messages to human messages if the LLM doesnt support system messages.
    """
    if not messages or not rules.require_system_message_first:
        return messages

    new_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            new_messages.append(HumanMessage(content=f"SYSTEM: {message.content}"))
        else:
            new_messages.append(message)
    return new_messages


def organize_tool_result_messages(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    if not messages or not rules.tool_result_must_follow_tool_call:
        return messages

    tool_calls = {}
    new_messages = []
    i = 0
    while i < len(messages):
        message = messages[i]
        # save the message index of any tool calls
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls + message.invalid_tool_calls:
                tool_calls[tool_call["id"]] = i
            new_messages.append(message)

        # move tool messages to follow their corresponding tool calls
        elif isinstance(message, ToolMessage) and tool_call["id"] in tool_calls:
            tool_call_index = tool_calls[tool_call["id"]]
            new_messages.insert(tool_call_index + 1, message)
            tool_calls[tool_call["id"]] += 1

        else:
            new_messages.append(message)
        i += 1
    return new_messages


def count_tokens(message: BaseMessage) -> int:
    # always use gpt-3.5 token counter with the entire message object; we only need to be approximate here
    return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(message.json()))


def trim_messages(
    messages: list[BaseMessage], max_tokens: Optional[int]
) -> list[BaseMessage]:
    """
    Trims messages to a maximum number of tokens while keeping the system message at the front.
    """

    if not messages or max_tokens is None:
        return messages

    new_messages = []
    budget = max_tokens

    for message in reversed(messages):
        if count_tokens(message) > budget:
            break
        new_messages.append(message)
        budget -= count_tokens(message)

    return list(reversed(new_messages))


@dataclass
class EventContext:
    llm_rules: LLMRules
    agent: Optional["Agent"]
    ready_tasks: list["Task"]
    flow: Optional["Flow"]
    controller: Optional["Controller"]


class MessageCompiler:
    def __init__(
        self,
        events: list[Event],
        context: EventContext,
        system_prompt: str = None,
        max_tokens: int = None,
    ):
        self.events = events
        self.context = context
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens or controlflow.settings.max_input_tokens

    def compile_to_messages(self) -> list[BaseMessage]:
        if self.system_prompt:
            system = [SystemMessage(content=self.system_prompt)]
            max_tokens = self.max_tokens - count_tokens(system[0])
        else:
            system = []
            max_tokens = self.max_tokens

        messages = []

        for event in self.events:
            messages.extend(event.to_messages(self.context))

        # process messages
        msgs = messages.copy()

        # trim messages
        msgs = trim_messages(msgs, max_tokens=max_tokens)

        # apply LLM rules
        msgs = ensure_at_least_one_message(msgs, rules=self.context.llm_rules)
        msgs = add_user_message_to_beginning(msgs, rules=self.context.llm_rules)
        msgs = add_user_message_to_end(msgs, rules=self.context.llm_rules)
        msgs = remove_duplicate_messages(msgs)
        msgs = organize_tool_result_messages(msgs, rules=self.context.llm_rules)
        msgs = break_up_consecutive_ai_messages(msgs, rules=self.context.llm_rules)

        # this should go last
        msgs = convert_system_messages(msgs, rules=self.context.llm_rules)

        return system + msgs
