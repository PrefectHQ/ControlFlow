import copy
import os
from contextlib import contextmanager
from pathlib import Path
from pyexpat import model
from typing import Any, Literal, Optional, Union

import prefect.logging.configuration
import prefect.settings
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from controlflow.utilities.general import unwrap

CONTROLFLOW_ENV_FILE = os.path.expanduser(
    os.path.expandvars(os.getenv("CONTROLFLOW_ENV_FILE", "~/.controlflow/.env"))
)


class ControlFlowSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CONTROLFLOW_",
        env_file=(
            "" if os.getenv("CONTROLFLOW_TEST_MODE") else (".env", CONTROLFLOW_ENV_FILE)
        ),
        extra="ignore",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class Settings(ControlFlowSettings):
    # ------------ home settings ------------

    home_path: Path = Field(
        default="~/.controlflow",
        description="The path to the ControlFlow home directory.",
        validate_default=True,
    )

    # ------------ display and logging settings ------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="The log level for ControlFlow.",
    )
    log_prints: bool = Field(
        default=False,
        description="Whether to log workflow prints to the Prefect logger by default.",
    )
    log_all_messages: bool = Field(
        default=False,
        description="If True, all LLM messages will be logged at the debug level.",
    )
    pretty_print_agent_events: Optional[bool] = Field(
        default=None,
        description="If True, agent events will be pretty-printed.",
        deprecated=True,
    )

    @model_validator(mode="before")
    def _validate_pretty_print_agent_events(cls, data: dict) -> dict:
        if data.get("pretty_print_agent_events") is not None:
            data["enable_default_print_handler"] = data["pretty_print_agent_events"]
            data["pretty_print_agent_events"] = None
            print(
                unwrap("""
                    The `pretty_print_agent_events` setting is deprecated, use
                    `enable_default_print_handler` instead. Your settings were
                    updated.
                    """)
            )
        return data

    enable_default_print_handler: bool = Field(
        default=True,
        description="If True, a PrintHandler will be enabled and automatically "
        "pretty-print agent events and completion tools.",
    )
    default_print_handler_show_completion_tools: bool = Field(
        default=True,
        description="If True, the default PrintHandler will include completion tools.",
    )
    default_print_handler_show_completion_tool_results: bool = Field(
        default=False,
        description="If True, the default PrintHandler will show the full results of completion tools.",
    )

    # ------------ orchestration settings ------------
    orchestrator_max_agent_turns: Optional[int] = Field(
        default=100,
        description="The default maximum number of agent turns per orchestration session."
        "If None, orchestration may run indefinitely. This setting can be overridden on a per-call basis.",
    )
    orchestrator_max_llm_calls: Optional[int] = Field(
        default=1000,
        description="The default maximum number of LLM calls per orchestrating session. "
        "If None, orchestration may run indefinitely. This setting can be overridden on a per-call basis.",
    )
    task_max_llm_calls: Optional[int] = Field(
        default=None,
        description="The default maximum number of LLM calls over a task's lifetime. "
        "If None, the task may run indefinitely. This setting can be overridden on a per-task basis.",
    )

    # ------------ LLM settings ------------

    llm_model: str = Field(
        default="openai/gpt-4o",
        description="The default LLM model for agents.",
    )
    llm_temperature: Union[float, None] = Field(
        None, description="The temperature for LLM sampling."
    )
    max_input_tokens: int = Field(
        100_000, description="The maximum number of tokens to send to an LLM."
    )

    # ------------ Memory settings ------------

    memory_provider: Optional[str] = Field(
        default="chroma-db",
        description="The default memory provider for agents.",
    )

    # ------------ Memory settings: ChromaDB ------------

    chroma_cloud_tenant: Optional[str] = Field(
        None,
        alias="CHROMA_CLOUD_TENANT",
        description="The tenant for Chroma Cloud.",
    )
    chroma_cloud_database: Optional[str] = Field(
        None,
        alias="CHROMA_CLOUD_DATABASE",
        description="The database for Chroma Cloud.",
    )
    chroma_cloud_api_key: Optional[str] = Field(
        None,
        alias="CHROMA_CLOUD_API_KEY",
        description="The API key for Chroma Cloud.",
    )

    # ------------ Debug settings ------------

    debug_messages: bool = Field(
        default=False,
        description="If True, all messages will be logged at the debug level.",
    )

    tools_raise_on_error: bool = Field(
        default=False,
        description="If True, an error in a tool call will raise an exception.",
    )

    tools_verbose: bool = Field(
        default=True, description="If True, tools will log additional information."
    )

    # ------------ experimental settings ------------

    enable_experimental_tui: bool = Field(
        default=False,
        description="If True, the experimental TUI will be enabled. If False, the TUI will be disabled.",
    )
    run_tui_headless: bool = Field(
        default=False,
        description="If True, the experimental TUI will run in headless mode, which is useful for debugging.",
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
    def set_log_level(self):
        from controlflow.utilities.logging import setup_logging

        setup_logging(level=self.log_level)
        return self

    @model_validator(mode="after")
    def _apply_prefect_settings(self):
        """
        Prefect settings are set at runtime by opening a settings context.
        We check if any prefect-specific settings have been changed and apply them.
        """
        if self._prefect_context is not None:
            self._prefect_context.__exit__(None, None, None)
            self._prefect_context = None

        settings_map = {
            "prefect_log_level": prefect.settings.PREFECT_LOGGING_LEVEL,
        }

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

        with temporary_settings(tools_raise_on_error=True):
            assert controlflow.settings.tools_raise_on_error is True
        assert controlflow.settings.tools_raise_on_error is False
        ```
    """
    old_settings = copy.deepcopy(settings.model_dump(exclude={"_prefect_context"}))

    try:
        # apply the new settings
        for attr, value in kwargs.items():
            if not hasattr(settings, attr):
                raise AttributeError(f"Setting {attr} does not exist.")
            setattr(settings, attr, value)
        yield

    finally:
        # restore the old settings
        for attr in kwargs:
            if hasattr(settings, attr):
                setattr(settings, attr, old_settings[attr])
