from langchain_core.language_models import BaseChatModel

import controlflow


def model_from_string(model: str) -> BaseChatModel:
    if "/" not in model:
        provider, model = "openai", model
    provider, model = model.split("/")

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "To use OpenAI models, please install the `langchain-openai` package."
            )
        cls = ChatOpenAI
    elif provider == "azure_openai":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(
                "To use Azure OpenAI models, please install the `langchain-openai` package."
            )
        cls = AzureChatOpenAI
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "To use Anthropic models, please install the `langchain-anthropic` package."
            )
        cls = ChatAnthropic
    else:
        raise ValueError(
            f"Could not load provider automatically: {provider}. Please create your model manually."
        )

    return cls(model=model)


def get_default_model() -> BaseChatModel:
    return model_from_string(model=controlflow.settings.llm_model)
