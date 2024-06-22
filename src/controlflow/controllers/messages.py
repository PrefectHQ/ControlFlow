from typing import Union

from controlflow.llm.messages import AIMessage, HumanMessage, MessageType, SystemMessage
from controlflow.llm.rules import LLMRules
from controlflow.llm.tools import Tool


def system_message(content: str, rules: LLMRules) -> Union[SystemMessage, HumanMessage]:
    if rules.system_message_must_be_first:
        return SystemMessage(content=content)
    else:
        return HumanMessage(content=f"SYSTEM: {content}")


def add_agent_info_to_messages(
    messages: list[MessageType], llm_rules: LLMRules
) -> list[MessageType]:
    """
    If the message is from an agent, add a system message to clarify which agent
    it is from. This helps the system follow multi-agent conversations.
    """
    new_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.agent:
            if llm_rules.system_message_must_be_first:
                system_msg = HumanMessage(
                    content=f'SYSTEM: The following message is from agent "{msg.agent.name}" with id {msg.agent.id}.'
                )
            else:
                system_msg = SystemMessage(
                    content=f'The following message is from agent "{msg.agent.name}" with id {msg.agent.id}.'
                )
            new_messages.append(system_msg)
        new_messages.append(msg)
    return new_messages


def process_messages(
    messages: list[MessageType],
    rules: LLMRules,
    tools: list[Tool],
):
    messages = messages.copy()

    messages = add_agent_info_to_messages(messages, rules)

    if rules.always_start_with_user_message:
        if not messages or not isinstance(messages[0], HumanMessage):
            messages.insert(0, HumanMessage(content="SYSTEM: Begin."))

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

    return messages
