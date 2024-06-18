import controlflow
from controlflow.settings import temporary_settings
from prefect.logging import get_logger


def test_temporary_settings():
    assert controlflow.settings.raise_on_tool_error is False
    with temporary_settings(raise_on_tool_error=True):
        assert controlflow.settings.raise_on_tool_error is True
    assert controlflow.settings.raise_on_tool_error is False


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
