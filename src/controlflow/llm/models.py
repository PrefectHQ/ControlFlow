import inspect
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

import controlflow
from controlflow.utilities.logging import get_logger

logger = get_logger(__name__)


def get_default_model() -> BaseChatModel:
    if getattr(controlflow.defaults, "model", None) is None:
        return model_from_string(controlflow.settings.llm_model)
    else:
        return controlflow.defaults.model


def model_from_string(
    model: str, temperature: Optional[float] = None, **kwargs: Any
) -> BaseChatModel:
    if "/" not in model:
        provider, model = "openai", model
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
                "To use Google models, please install the `langchain_google_genai` package."
            )
        cls = ChatGoogleGenerativeAI
    elif provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "To use Groq models, please install the `langchain_groq` package."
            )
        cls = ChatGroq
    else:
        raise ValueError(
            f"Could not load provider automatically: {provider}. Please create your model manually."
        )

    return cls(model=model, temperature=temperature, **kwargs)


def _get_initial_default_model() -> BaseChatModel:
    # special error messages for the initial attempt to create a model
    try:
        return model_from_string(controlflow.settings.llm_model)
    except Exception as exc:
        if isinstance(exc, ValidationError) and "Did not find openai_api_key" in str(
            exc
        ):
            msg = inspect.cleandoc("""
                The default LLM model could not be created because the OpenAI
                API key was not found. ControlFlow will continue to work, but
                you must manually provide an LLM model for each agent. Please
                set the OPENAI_API_KEY environment variable or choose a
                different default LLM model. For more information, please see
                https://controlflow.ai/guides/llms.
                """).replace("\n", " ")
        else:
            msg = (
                inspect.cleandoc("""
                The default LLM model could not be created. ControlFlow will
                continue to work, but you must manually provide an LLM model for
                each agent. For more information, please see
                https://controlflow.ai/guides/llms. The error was:
                """).replace("\n", " ")
                + f"\n{exc}"
            )
        logger.warn(msg)
