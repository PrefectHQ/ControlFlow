import inspect
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

import controlflow
from controlflow.utilities.general import unwrap
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


def get_default_model() -> BaseChatModel:
    if getattr(controlflow.defaults, "model", None) is None:
        return get_model(controlflow.settings.llm_model)
    else:
        return controlflow.defaults.model


def get_model(
    model: str, temperature: Optional[float] = None, **kwargs: Any
) -> BaseChatModel:
    """Get a model from a string."""
    if "/" not in model:
        raise ValueError(
            f"The model `{model}` is not valid. Please specify the provider "
            "and model name, e.g. 'openai/gpt-4o-mini' or 'anthropic/claude-3-haiku-20240307'."
        )
    provider, model = model.split("/")

    if temperature is None:
        temperature = controlflow.settings.llm_temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        cls = ChatOpenAI
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        cls = ChatAnthropic
    elif provider == "azure-openai":
        from langchain_openai import AzureChatOpenAI

        cls = AzureChatOpenAI
    elif provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "To use Google as an LLM provider, please install the `langchain_google_genai` package."
            )
        cls = ChatGoogleGenerativeAI
    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "To use Groq as an LLM provider, please install the `langchain_groq` package."
            )
        cls = ChatGroq
    else:
        raise ValueError(
            f"Could not load provider `{provider}` automatically. Please provide the LLM class manually."
        )

    return cls(model=model, temperature=temperature, **kwargs)


def _get_initial_default_model() -> BaseChatModel:
    # special error messages for the initial attempt to create a model
    try:
        return get_model(controlflow.settings.llm_model)
    except Exception as exc:
        if isinstance(exc, ValidationError) and "Did not find openai_api_key" in str(
            exc
        ):
            msg = unwrap("""
                The default LLM model could not be created because the OpenAI
                API key was not found. ControlFlow will continue to work, but
                you must manually provide an LLM model for each agent. Please
                set the OPENAI_API_KEY environment variable or choose a
                different default LLM model. For more information, please see
                https://controlflow.ai/guides/configure-llms.
                """).replace("\n", " ")
        else:
            msg = (
                unwrap("""
                The default LLM model could not be created. ControlFlow will
                continue to work, but you must manually provide an LLM model for
                each agent. For more information, please see
                https://controlflow.ai/guides/configure-llms. The error was:
                """).replace("\n", " ")
                + f"\n{exc}"
            )
        logger.warning(msg)
