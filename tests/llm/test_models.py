import pytest
import pytest_mock
from controlflow.llm.models import (
    get_default_model,
    get_model_from_string,
    get_provider_from_string,
)


def test_get_provider_from_string_openai():
    provider = get_provider_from_string("openai")
    assert provider.__name__ == "ChatOpenAI"

def test_get_provider_from_string_azure_openai():
    provider = get_provider_from_string("azure_openai")
    assert provider.__name__ == "AzureChatOpenAI"

def test_get_provider_from_string_anthropic():
    pytest.importorskip("langchain_anthropic")
    provider = get_provider_from_string("anthropic")
    assert provider.__name__ == "ChatAnthropic"

def test_get_provider_from_string_google():
    pytest.importorskip("langchain_google_genai")
    provider = get_provider_from_string("google")
    assert provider.__name__ == "ChatGoogleGenerativeAI"

def test_get_provider_from_string_invalid_provider():
    with pytest.raises(ValueError):
        get_provider_from_string("invalid_provider.gpt4")

def test_get_provider_from_string_missing_module():
    with pytest.raises(ImportError):
        get_provider_from_string("openai.missing_module")

def test_get_model_from_string(mocker: pytest_mock.MockFixture):
    # Test getting a model from string
    mock_provider_class = mocker.Mock()
    mock_provider_instance = mocker.Mock()
    mock_provider_class.return_value = mock_provider_instance
    mocker.patch("controlflow.llm.models.get_provider_from_string", return_value=mock_provider_class)
    model = get_model_from_string("openai/davinci", temperature=0.5)
    assert model == mock_provider_instance
    mock_provider_class.assert_called_once_with(
        name="davinci",
        temperature=0.5,
    )

    # Test getting a model with default settings
    mock_provider_class.reset_mock()
    mocker.patch("controlflow.settings.settings.llm_model", "anthropic/claude")
    mocker.patch("controlflow.settings.settings.llm_temperature", 0.7)
    pytest.importorskip("langchain_anthropic")  # Skip if langchain_anthropic is not installed
    model = get_model_from_string()
    assert model == mock_provider_instance
    mock_provider_class.assert_called_once_with(
        name="claude",
        temperature=0.7,
    )

def test_get_default_model(mocker: pytest_mock.MockFixture):
    # Test getting the default model
    mock_get_model_from_string = mocker.patch("controlflow.llm.models.get_model_from_string")
    get_default_model()
    mock_get_model_from_string.assert_called_once_with()