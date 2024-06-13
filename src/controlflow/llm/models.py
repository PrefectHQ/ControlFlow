from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional, Union

from langchain_core.language_models import BaseChatModel

import controlflow

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

_model_registry: dict[str, tuple[str, str]] = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
}


def get_provider_from_string(
    provider: str,
) -> Union[
    type["ChatOpenAI"],
    type["AzureChatOpenAI"],
    type["ChatAnthropic"],
    type["ChatGoogleGenerativeAI"],
]:
    module_name, class_name = _model_registry.get(provider, ("openai", ""))
    if not class_name:
        raise ValueError(
            f"Could not load provider automatically: {provider}. Please create your model manually."
        )
    try:
        module = import_module(module_name)
    except ImportError:
        raise ImportError(
            f"To use {provider} models, please install the `{module_name}` package."
        )
    return getattr(module, class_name)  # type: ignore[no-any-return]


def get_model_from_string(
    model: Optional[str] = None, temperature: Optional[float] = None, **kwargs: Any
) -> BaseChatModel:
    provider, _, model = (model or controlflow.settings.llm_model).partition("/")
    return get_provider_from_string(provider=provider)(
        name=model or controlflow.settings.llm_model,
        temperature=temperature or controlflow.settings.llm_temperature,
        **kwargs,
    )


def get_default_model() -> BaseChatModel:
    if controlflow.default_model is None:
        return get_model_from_string(controlflow.settings.llm_model)
    else:
        return controlflow.default_model


DEFAULT_MODEL = None
