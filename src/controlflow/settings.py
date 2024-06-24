import copy
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import prefect.logging.configuration
import prefect.settings
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    pass


class ControlFlowSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONTROLFLOW_",
        env_file=(
            ""
            if os.getenv("CONTROLFLOW_TEST_MODE")
            else ("~/.controlflow/.env", ".env")
        ),
        extra="ignore",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class Settings(ControlFlowSettings):
    max_task_iterations: Optional[int] = Field(
        default=100,
        description="The maximum number of iterations to attempt to complete a task "
        "before raising an error. If None, the task will run indefinitely. "
        "This setting can be overridden by the `max_iterations` attribute "
        "on a task.",
    )

    # ------------ home settings ------------

    home_path: Path = Field(
        default="~/.controlflow",
        description="The path to the ControlFlow home directory.",
        validate_default=True,
    )

    # ------------ display and logging settings ------------

    log_prints: bool = Field(
        False,
        description="Whether to log workflow prints to the Prefect logger by default.",
    )

    # ------------ flow settings ------------

    eager_mode: bool = Field(
        default=True,
        description="If True, @task- and @flow-decorated functions are run immediately. "
        "This can be set on a per-task or per-flow basis using the `eager` argument.",
    )
    enable_local_input: bool = Field(
        default=True,
        description="If True, the user can provide input via "
        "the terminal. Otherwise, only API input is accepted.",
    )
    strict_flow_context: bool = Field(
        default=False,
        description="If False, calling Task.run() outside a flow context will automatically "
        "create a flow and run the task within it. If True, an error will be raised.",
    )

    # ------------ LLM settings ------------

    llm_model: str = Field(default="openai/gpt-4o", description="The LLM model to use.")
    llm_temperature: float = Field(0.7, description="The temperature for LLM sampling.")
    max_input_tokens: int = Field(
        100_000, description="The maximum number of tokens to send to an LLM."
    )

    # ------------ Flow visualization settings ------------

    enable_print_handler: bool = Field(
        default=True,
        description="If True, the PrintHandler will be enabled. Superseded by the enable_tui setting.",
    )

    enable_tui: bool = Field(
        default=False,
        description="If True, the TUI will be enabled. If False, the TUI will be disabled.",
    )
    run_tui_headless: bool = Field(
        default=False,
        description="If True, the TUI will run in headless mode, which is useful for debugging.",
    )

    # ------------ Debug settings ------------

    raise_on_tool_error: bool = Field(
        False, description="If True, an error in a tool call will raise an exception."
    )

    # ------------ Prefect settings ------------
    #
    # Default settings for Prefect when used with ControlFlow. They can be
    # overridden by setting standard Prefect env vars

    prefect_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="WARNING",
        description="The log level for Prefect.",
        alias="PREFECT_LOGGING_LEVEL",
    )

    _prefect_context: contextmanager = None

    @field_validator("home_path", mode="before")
    def _validate_home_path(cls, v: Union[str, Path]) -> Path:
        v = Path(v).expanduser()
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def _apply_prefect_settings(self):
        """
        Prefect settings are set at runtime by opening a settings context.
        We check if any prefect-specific settings have been changed and apply them.
        """
        if self._prefect_context is not None:
            self._prefect_context.__exit__(None, None, None)
            self._prefect_context = None

        settings_map = {"prefect_log_level": prefect.settings.PREFECT_LOGGING_LEVEL}

        prefect_settings = {}

        for cf_setting, v in self.model_dump().items():
            if cf_setting.startswith("prefect_"):
                p_setting = settings_map[cf_setting]
                if p_setting.value() != v:
                    prefect_settings[settings_map[cf_setting]] = v

        if prefect_settings:
            self._prefect_context = prefect.settings.temporary_settings(
                prefect_settings
            )
            self._prefect_context.__enter__()

        # Configure logging
        prefect.logging.configuration.setup_logging()

        return self


settings = Settings()


@contextmanager
def temporary_settings(**kwargs: Any):
    """
    Temporarily override ControlFlow setting values.

    Args:
        **kwargs: The settings to override, including nested settings.

    Example:
        Temporarily override a setting:
        ```python
        import controlflow
        from controlflow.settings import temporary_settings

        with temporary_settings(raise_on_tool_error=True):
            assert controlflow.settings.raise_on_tool_error is True
        assert controlflow.settings.raise_on_tool_error is False
        ```
    """
    old_settings = copy.deepcopy(settings.model_dump(exclude={"_prefect_context"}))

    try:
        # apply the new settings
        for attr, value in kwargs.items():
            setattr(settings, attr, value)
        yield

    finally:
        # restore the old settings
        for attr in kwargs:
            setattr(settings, attr, old_settings[attr])
