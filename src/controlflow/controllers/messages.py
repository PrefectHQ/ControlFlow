from typing import Optional, Union

from controlflow.agents.agent import Agent
from controlflow.llm.messages import (
    AIMessage,
    HumanMessage,
    MessageType,
    SystemMessage,
    ToolMessage,
)
from controlflow.llm.rules import LLMRules
from controlflow.llm.tools import Tool


def system_message(content: str, rules: LLMRules) -> Union[SystemMessage, HumanMessage]:
    if rules.system_message_must_be_first:
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=f"SYSTEM: {content}")


def add_agent_info_to_messages(messages: list[MessageType]) -> list[MessageType]:
    """
    If the message is from an agent, add a system message to clarify which agent
    it is from. This helps the system follow multi-agent conversations.
    """
    new_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.agent:
            system_msg = SystemMessage(
                content=f'The following message is from agent "{msg.agent.name}" with id {msg.agent.id}.'
            )
            new_messages.append(system_msg)
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
                msg = HumanMessage(content=f"SYSTEM: {msg.content}")
            new_messages.append(msg)

        return new_messages
    else:
        return messages


def handle_user_message_must_be_first_after_system(
    messages: list[MessageType], rules: LLMRules
):
    if rules.user_message_must_be_first_after_system:
        if not messages:
            messages.append(HumanMessage(content="SYSTEM: Begin."))

        # else get first non-system message
        else:
            i = 0
            while i < len(messages) and isinstance(messages[i], SystemMessage):
                i += 1
            if i == len(messages) or (
                i < len(messages) and not isinstance(messages[i], HumanMessage)
            ):
                messages.insert(i, HumanMessage(content="SYSTEM: Begin."))
    return messages


def handle_private_tool_calls(messages: list[MessageType], agent: Agent):
    new_messages = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if agent.id != msg.agent_id:
                msg = ToolMessage(
                    content=f"The result of this tool call only visible to agent {msg.agent_id}",
                    tool_call_id=msg.tool_call_id,
                    tool_metadata=msg.tool_metadata | {"is_private": True},
                )
        new_messages.append(msg)
    return new_messages


def prepare_messages(
    agent: Agent,
    messages: list[MessageType],
    system_message: Optional[SystemMessage],
    rules: LLMRules,
    tools: list[Tool],
):
    messages = messages.copy()

    if system_message is not None:
        messages.insert(0, system_message)

    messages = add_agent_info_to_messages(messages)

    if not rules.allow_last_message_has_ai_role_with_tools:
        if messages and tools and isinstance(messages[-1], AIMessage):
            messages.append(system_message("Continue.", rules))

    if not rules.allow_consecutive_ai_messages:
        if messages:
            i = 1
            while i < len(messages):
                if isinstance(messages[i], AIMessage) and isinstance(
                    messages[i - 1], AIMessage
                ):
                    messages.insert(i, system_message("Continue.", rules))
                    i += 1

    messages = handle_system_messages_must_be_first(messages, rules)
    messages = handle_user_message_must_be_first_after_system(messages, rules)
    messages = handle_private_tool_calls(messages, agent)

    return messages
