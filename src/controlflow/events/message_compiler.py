import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import tiktoken

import controlflow
from controlflow.events.base import Event, UnpersistedEvent
from controlflow.events.events import (
    AgentMessage,
    ToolCallEvent,
    ToolResultEvent,
)
from controlflow.llm.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from controlflow.llm.rules import LLMRules
from controlflow.utilities.logging import get_logger

if TYPE_CHECKING:
    from controlflow.agents.agent import Agent
logger = get_logger(__name__)


class CombinedAgentMessage(UnpersistedEvent):
    event: Literal["combined-agent-message"] = "combined-agent-message"
    agent_message: AgentMessage
    tool_call: list[ToolCallEvent] = []
    tool_results: list[ToolResultEvent] = []

    def to_messages(self, context: "CompileContext") -> list[BaseMessage]:
        messages = []
        messages.extend(self.agent_message.to_messages(context))
        for tool_result in self.tool_results:
            messages.extend(tool_result.to_messages(context))
        return messages


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
    Converts system messages to human messages if the LLM doesnt support system
    messages, either at all or in the first position.
    """
    new_messages = []
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            # If system messages are not supported OR if they must be first and
            # this is not the first message, THEN convert the message to a human message
            if not rules.allow_system_messages or (
                i > 0 and rules.require_system_message_first
            ):
                new_messages.append(
                    HumanMessage(
                        content=f"ORCHESTRATOR: {message.content}", name=message.name
                    )
                )
            else:
                # If the system message is allowed, add it as-is
                new_messages.append(message)
        else:
            new_messages.append(message)
    return new_messages


def format_message_name(
    messages: list[BaseMessage], rules: LLMRules
) -> list[BaseMessage]:
    if not rules.require_message_name_format:
        return messages

    for message in messages:
        if message.name:
            name = re.sub(rules.require_message_name_format, "-", message.name)
            message.name = name.strip("-")
    return messages


def count_tokens(message: BaseMessage) -> int:
    # always use gpt-3.5 token counter with the entire message object; we only need to be approximate here
    return len(
        tiktoken.encoding_for_model("gpt-3.5-turbo").encode(message.model_dump_json())
    )


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
class CompileContext:
    llm_rules: LLMRules
    agent: Optional["Agent"]


class MessageCompiler:
    def __init__(
        self,
        events: list[Event],
        system_prompt: Optional[str] = None,
        llm_rules: Optional[LLMRules] = None,
        max_tokens: Optional[int] = None,
    ):
        self.events = events
        self.system_prompt = system_prompt
        self.llm_rules = llm_rules
        self.max_tokens = max_tokens or controlflow.settings.max_input_tokens

    def organize_events(self, context: CompileContext) -> list[Event]:
        organized_events = []
        tool_calls = {}

        for event in self.events:
            # combine all agent messages and tool results
            if isinstance(event, AgentMessage):
                # add a combined agent message
                combined_event = CombinedAgentMessage(agent_message=event)
                organized_events.append(combined_event)
                # register the combined message under each tool call id
                for tc in (
                    event.ai_message.tool_calls + event.ai_message.invalid_tool_calls
                ):
                    tool_calls[tc["id"]] = combined_event
            elif isinstance(event, ToolResultEvent):
                combined_event: CombinedAgentMessage = tool_calls.get(
                    event.tool_call["id"]
                )
                if combined_event:
                    combined_event.tool_results.append(event)

            # all other events are added as-is
            else:
                organized_events.append(event)

        return organized_events

    def compile_to_messages(self, agent: "Agent") -> list[BaseMessage]:
        context = CompileContext(
            agent=agent, llm_rules=self.llm_rules or agent.get_llm_rules()
        )

        if self.system_prompt:
            system_prompt = [SystemMessage(content=self.system_prompt)]
            max_tokens = self.max_tokens - count_tokens(system_prompt[0])
        else:
            system_prompt = []
            max_tokens = self.max_tokens

        events = self.organize_events(context=context)

        messages = []
        for event in events:
            messages.extend(event.to_messages(context))

        # trim messages
        messages = trim_messages(messages, max_tokens=max_tokens)

        # apply LLM rules
        messages = ensure_at_least_one_message(messages, rules=context.llm_rules)
        messages = add_user_message_to_beginning(messages, rules=context.llm_rules)
        messages = add_user_message_to_end(messages, rules=context.llm_rules)
        messages = remove_duplicate_messages(messages)
        messages = break_up_consecutive_ai_messages(messages, rules=context.llm_rules)
        messages = format_message_name(messages, rules=context.llm_rules)

        messages = system_prompt + messages

        # this should go last
        messages = convert_system_messages(messages, rules=context.llm_rules)

        return messages
