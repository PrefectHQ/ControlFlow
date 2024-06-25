import importlib

import controlflow
import pytest
from controlflow.settings import temporary_settings
from prefect.logging import get_logger


def test_temporary_settings():
    assert controlflow.settings.tools_raise_on_error is False
    with temporary_settings(tools_raise_on_error=True):
        assert controlflow.settings.tools_raise_on_error is True
    assert controlflow.settings.tools_raise_on_error is False


def test_prefect_settings_apply_at_runtime(caplog):
    prefect_logger = get_logger()
    assert controlflow.settings.prefect_log_level == "WARNING"
    prefect_logger.warning("test-log-1")
    controlflow.settings.prefect_log_level = "ERROR"
    prefect_logger.warning("test-log-2")
    controlflow.settings.prefect_log_level = "DEBUG"
    prefect_logger.warning("test-log-3")

    assert "test-log-1" in caplog.text
    assert "test-log-2" not in caplog.text
    assert "test-log-3" in caplog.text


def test_import_without_default_api_key_warns_but_does_not_fail(monkeypatch, caplog):
    # remove the OPENAI_API_KEY environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Clear any previous logs
    caplog.clear()

    # Import the library
    with caplog.at_level("WARNING"):
        # Reload the library to apply changes
        importlib.reload(controlflow)

    # Check if the warning was logged
    assert any(
        record.levelname == "WARNING"
        and "The default LLM model could not be created" in record.message
        for record in caplog.records
    ), "The expected warning was not logged"


def test_import_without_default_api_key_errors_when_loading_model(monkeypatch):
    # remove the OPENAI_API_KEY environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Reload the library to apply changes
    importlib.reload(controlflow)

    with pytest.raises(ValueError, match="Did not find openai_api_key"):
        controlflow.get_default_model()

    with pytest.raises(
        ValueError, match="No model provided and no default model could be loaded"
    ):
        controlflow.Agent().get_model()


def test_import_without_api_key_for_non_default_model_warns_but_does_not_fail(
    monkeypatch, caplog
):
    # remove the OPENAI_API_KEY environment variable
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CONTROLFLOW_LLM_MODEL", "anthropic/not-a-model")

    # Clear any previous logs
    caplog.clear()

    # Import the library
    with caplog.at_level("WARNING"):
        # Reload the library to apply changes
        importlib.reload(controlflow)

    # Check if the warning was logged
    assert any(
        record.levelname == "WARNING"
        and "The default LLM model could not be created" in record.message
        for record in caplog.records
    ), "The expected warning was not logged"
