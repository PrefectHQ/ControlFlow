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

    # messages
    always_start_with_user_message: bool = False
    allow_last_message_has_ai_role_with_tools: bool = True
    system_message_must_be_first: bool = False
    allow_consecutive_ai_messages: bool = True


class OpenAIRules(LLMRules):
    pass


class AnthropicRules(LLMRules):
    always_start_with_user_message: bool = True
    system_message_must_be_first: bool = True
    allow_last_message_has_ai_role_with_tools: bool = False
    allow_consecutive_ai_messages: bool = False


def rules_for_model(model: BaseChatModel) -> LLMRules:
    if isinstance(model, (ChatOpenAI, AzureChatOpenAI)):
        return OpenAIRules()
    elif isinstance(model, ChatAnthropic):
        return AnthropicRules()
    else:
        return LLMRules()
