from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

import controlflow


def get_default_model() -> BaseChatModel:
    if getattr(controlflow, "default_model", None) is None:
        return model_from_string(controlflow.settings.llm_model)
    else:
        return controlflow.default_model


def model_from_string(
    model: str, temperature: Optional[float] = None, **kwargs: Any
) -> BaseChatModel:
    if "/" not in model:
        provider, model = "openai", model
    provider, model = model.split("/")

    if temperature is None:
        temperature = controlflow.settings.llm_temperature

    if provider == "openai":
        cls = ChatOpenAI
    elif provider == "anthropic":
        cls = ChatAnthropic
    elif provider == "azure-openai":
        from langchain_openai import AzureChatOpenAI

        cls = AzureChatOpenAI
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "To use Google models, please install the `langchain_google_genai` package."
            )
        cls = ChatGoogleGenerativeAI
    else:
        raise ValueError(
            f"Could not load provider automatically: {provider}. Please create your model manually."
        )

    return cls(model=model, temperature=temperature, **kwargs)


DEFAULT_MODEL = None
