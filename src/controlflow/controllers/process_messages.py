from typing import Optional, Union

from controlflow.agents.agent import Agent
from controlflow.llm.messages import (
    AIMessage,
    MessageType,
    SystemMessage,
    UserMessage,
)
from controlflow.llm.rules import LLMRules
from controlflow.tools import Tool


def create_system_message(
    content: str, rules: LLMRules
) -> Union[SystemMessage, UserMessage]:
    """
    Creates a SystemMessage or HumanMessage with SYSTEM: prefix, depending on the rules.
    """
    if rules.system_message_must_be_first:
        return SystemMessage(content=content)
    else:
        return UserMessage(content=f"SYSTEM: {content}")


def handle_agent_info_in_messages(
    messages: list[MessageType], agent: Agent, rules: LLMRules
) -> list[MessageType]:
    """
    If the message is from an agent, add a system message immediately before it to clarify which agent
    it is from. This helps the system follow multi-agent conversations.

    """
    if not rules.add_system_messages_for_multi_agent:
        return messages

    current_agent = agent
    new_messages = []
    for msg in messages:
        # if the message is from a different agent than the previous message,
        # add a clarifying system message
        if isinstance(msg, AIMessage) and msg.agent and msg.agent != current_agent:
            system_msg = SystemMessage(
                content=f'The following message is from agent "{msg.agent["name"]}" '
                f'with id {msg.agent["id"]}.'
            )
            new_messages.append(system_msg)
            current_agent = msg.agent
        new_messages.append(msg)
    return new_messages


def handle_system_messages_must_be_first(messages: list[MessageType], rules: LLMRules):
    if rules.system_message_must_be_first:
        new_messages = []
        # consolidate consecutive SystemMessages into one
        if isinstance(messages[0], SystemMessage):
            content = [messages[0].content]
            i = 1
            while i < len(messages) and isinstance(messages[i], SystemMessage):
                i += 1
                content.append(messages[i].content)
            new_messages.append(SystemMessage(content="\n\n".join(content)))

        # replace all other SystemMessages with HumanMessages
        for i, msg in enumerate(messages[len(new_messages) :]):
            if isinstance(msg, SystemMessage):
                msg = UserMessage(content=f"SYSTEM: {msg.content}")
            new_messages.append(msg)

        return new_messages
    else:
        return messages


def handle_user_message_must_be_first_after_system(
    messages: list[MessageType], rules: LLMRules
):
    if rules.user_message_must_be_first_after_system:
        if not messages:
            messages.append(UserMessage(content="SYSTEM: Begin."))

        # else get first non-system message
        else:
            i = 0
            while i < len(messages) and isinstance(messages[i], SystemMessage):
                i += 1
            if i == len(messages) or (
                i < len(messages) and not isinstance(messages[i], UserMessage)
            ):
                messages.insert(i, UserMessage(content="SYSTEM: Begin."))
    return messages


def prepare_messages(
    agent: Agent,
    messages: list[MessageType],
    system_message: Optional[SystemMessage],
    rules: LLMRules,
    tools: list[Tool],
):
    """This is the main function for processing messages. It applies all the rules"""
    messages = messages.copy()

    if system_message is not None:
        messages.insert(0, system_message)

    messages = handle_agent_info_in_messages(messages, agent=agent, rules=rules)

    if not rules.allow_last_message_has_ai_role_with_tools:
        if messages and tools and isinstance(messages[-1], AIMessage):
            messages.append(create_system_message("Continue.", rules=rules))

    if not rules.allow_consecutive_ai_messages:
        if messages:
            i = 1
            while i < len(messages):
                if isinstance(messages[i], AIMessage) and isinstance(
                    messages[i - 1], AIMessage
                ):
                    messages.insert(i, create_system_message("Continue.", rules=rules))
                i += 1

    messages = handle_system_messages_must_be_first(messages, rules=rules)
    messages = handle_user_message_must_be_first_after_system(messages, rules=rules)

    return messages
