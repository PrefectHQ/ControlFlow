import importlib

import openai
import pytest
from prefect.logging import get_logger

import controlflow
import controlflow.llm.models
from controlflow.settings import temporary_settings


def test_defaults():
    # ensure that debug settings etc. are not misconfigured during development
    # change these settings to match whatever the default should be
    assert controlflow.settings.tools_raise_on_error is False
    assert controlflow.settings.tools_verbose is True
    # 4o is the default
    assert controlflow.settings.llm_model == "openai/gpt-4o-mini"
    assert controlflow.settings.prefect_log_level == "DEBUG"


def test_temporary_settings():
    assert controlflow.settings.tools_raise_on_error is False
    with temporary_settings(tools_raise_on_error=True):
        assert controlflow.settings.tools_raise_on_error is True
    assert controlflow.settings.tools_raise_on_error is False


def test_prefect_settings_apply_at_runtime(caplog):
    prefect_logger = get_logger()
    controlflow.settings.prefect_log_level = "WARNING"
    prefect_logger.warning("test-log-1")
    controlflow.settings.prefect_log_level = "ERROR"
    prefect_logger.warning("test-log-2")
    controlflow.settings.prefect_log_level = "DEBUG"
    prefect_logger.warning("test-log-3")

    assert "test-log-1" in caplog.text
    assert "test-log-2" not in caplog.text
    assert "test-log-3" in caplog.text


def test_import_without_default_api_key_warns_but_does_not_fail(monkeypatch, caplog):
    try:
        with monkeypatch.context() as m:
            # remove the OPENAI_API_KEY environment variable
            m.delenv("OPENAI_API_KEY", raising=False)

            # Clear any previous logs
            caplog.clear()

            # Import the library
            with caplog.at_level("DEBUG"):
                # Reload the library to apply changes
                defaults_module = importlib.import_module("controlflow.defaults")
                importlib.reload(defaults_module)
                importlib.reload(controlflow)

            # Check if the warning was logged
            assert any(
                record.levelname == "WARNING"
                and "The default LLM model could not be created" in record.message
                for record in caplog.records
            ), "The expected warning was not logged"
    finally:
        defaults_module = importlib.import_module("controlflow.defaults")
        importlib.reload(defaults_module)
        importlib.reload(controlflow)


def test_import_without_default_api_key_errors_when_loading_model(monkeypatch):
    try:
        with monkeypatch.context() as m:
            # remove the OPENAI_API_KEY environment variable
            m.delenv("OPENAI_API_KEY", raising=False)

            # Reload the library to apply changes
            defaults_module = importlib.import_module("controlflow.defaults")
            importlib.reload(defaults_module)
            importlib.reload(controlflow)

            with pytest.raises(
                openai.OpenAIError, match="api_key client option must be set"
            ):
                controlflow.llm.models.get_default_model()

            with pytest.raises(
                ValueError,
                match="No model provided and no default model could be loaded",
            ):
                controlflow.Agent().get_model()
    finally:
        defaults_module = importlib.import_module("controlflow.defaults")
        importlib.reload(defaults_module)
        importlib.reload(controlflow)


def test_import_without_api_key_for_non_default_model_warns_but_does_not_fail(
    monkeypatch, caplog
):
    try:
        with monkeypatch.context() as m:
            # remove the OPENAI_API_KEY environment variable
            m.delenv("OPENAI_API_KEY", raising=False)
            m.setenv("CONTROLFLOW_LLM_MODEL", "anthropic/not-a-model")

            # Clear any previous logs
            caplog.clear()

            # Import the library
            with caplog.at_level("WARNING"):
                # Reload the library to apply changes
                defaults_module = importlib.import_module("controlflow.defaults")
                importlib.reload(defaults_module)
                importlib.reload(controlflow)

            # Check if the warning was logged
            assert any(
                record.levelname == "WARNING"
                and "The default LLM model could not be created" in record.message
                for record in caplog.records
            ), "The expected warning was not logged"
    finally:
        defaults_module = importlib.import_module("controlflow.defaults")
        importlib.reload(defaults_module)
        importlib.reload(controlflow)
