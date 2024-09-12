import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from controlflow.llm.models import get_model


def test_get_model_from_openai():
    model = get_model("openai/gpt-4o-mini")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"


def test_get_model_from_anthropic():
    model = get_model("anthropic/claude-3-haiku-20240307")
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-3-haiku-20240307"


def test_get_azure_openai_model():
    model = get_model("azure-openai/gpt-4")
    assert isinstance(model, AzureChatOpenAI)
    assert model.deployment_name == "gpt-4"


def test_get_google_model():
    model = get_model("google/gemini-1.5-pro")
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "models/gemini-1.5-pro"


def test_get_groq_model():
    model = get_model("groq/mixtral-8x7b-32768")
    assert isinstance(model, ChatGroq)
    assert model.model_name == "mixtral-8x7b-32768"


def test_get_model_with_bad_format():
    with pytest.raises(ValueError, match="The model `xyz` is not valid."):
        get_model("xyz")


def test_get_model_with_unsupported_provider():
    with pytest.raises(
        ValueError, match="Could not load provider `unsupported` automatically."
    ):
        get_model("unsupported/model-name")


def test_get_model_with_temperature():
    model = get_model("anthropic/claude-3-haiku-20240307", temperature=0.7)
    assert isinstance(model, ChatAnthropic)
    assert model.temperature == 0.7
