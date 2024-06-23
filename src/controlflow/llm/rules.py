from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from controlflow.llm.models import BaseChatModel


class LLMRules:
    """
    LLM rules let us tailor DAG compilation, message generation, tool use, and
    other behavior to the requirements of different LLM provider APIs.

    Rules can be added here (to the base class) and overridden in subclasses, if
    necessary.
    """

    # system messages can only be provided as the very first message in a thread
    system_message_must_be_first: bool = False

    # other than a system message, the first message must be from the user
    user_message_must_be_first_after_system: bool = False

    # the last message in a thread can't be from an AI if tool use is allowed
    allow_last_message_has_ai_role_with_tools: bool = True

    # consecutive AI messages must be separated by a user message
    allow_consecutive_ai_messages: bool = True

    # add system messages to identify speakers in multi-agent conversations
    # (some APIs can use the `name` field for this purpose, but others can't)
    add_system_messages_for_multi_agent: bool = False


class OpenAIRules(LLMRules):
    pass


class AnthropicRules(LLMRules):
    system_message_must_be_first: bool = True
    user_message_must_be_first_after_system: bool = True
    allow_last_message_has_ai_role_with_tools: bool = False
    allow_consecutive_ai_messages: bool = False


def rules_for_model(model: BaseChatModel) -> LLMRules:
    if isinstance(model, (ChatOpenAI, AzureChatOpenAI)):
        return OpenAIRules()
    elif isinstance(model, ChatAnthropic):
        return AnthropicRules()
    else:
        return LLMRules()
