import textwrap
from typing import Any, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import Field

from controlflow.llm.models import BaseChatModel
from controlflow.utilities.general import ControlFlowModel, unwrap


class LLMRules(ControlFlowModel):
    """
    LLM rules let us tailor DAG compilation, message generation, tool use, and
    other behavior to the requirements of different LLM provider APIs.

    Rules can be added here (to the base class) and overridden in subclasses, if
    necessary.
    """

    model: Any

    # require at least one non-system message
    require_at_least_one_message: bool = False

    # system messages are supported as a role
    allow_system_messages: bool = True

    # system messages can only be provided as the very first message in a thread
    require_system_message_first: bool = False

    # other than a system message, the first message must be from the user
    require_user_message_after_system: bool = False

    # the last message in a thread can't be from an AI if tool use is allowed
    allow_last_message_from_ai_when_using_tools: bool = True

    # consecutive AI messages must be separated by a user message
    allow_consecutive_ai_messages: bool = True

    # add system messages to identify speakers in multi-agent conversations
    # (some APIs can use the `name` field for this purpose, but others can't)
    add_system_messages_for_multi_agent: bool = False

    # if a tool is used, the result must follow the tool call immediately
    tool_result_must_follow_tool_call: bool = True

    # the name associated with a message must conform to a specific format
    require_message_name_format: Optional[str] = None

    def model_instructions(self) -> Optional[list[str]]:
        pass


class OpenAIRules(LLMRules):
    require_message_name_format: str = r"[^a-zA-Z0-9_-]"

    model: Any

    def model_instructions(self) -> list[str]:
        instructions = []
        return instructions


class AnthropicRules(LLMRules):
    require_at_least_one_message: bool = True
    require_system_message_first: bool = True
    require_user_message_after_system: bool = True
    allow_last_message_from_ai_when_using_tools: bool = False
    allow_consecutive_ai_messages: bool = False


def rules_for_model(model: BaseChatModel) -> LLMRules:
    if isinstance(model, (ChatOpenAI, AzureChatOpenAI)):
        return OpenAIRules(model=model)
    if isinstance(model, ChatAnthropic):
        return AnthropicRules(model=model)

    try:
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        if isinstance(model, ChatAnthropicVertex):
            return AnthropicRules(model=model)
    except ImportError:
        pass

    # catchall
    return LLMRules(model=model)
