import os
import sys
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlFlowSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_prefix="CONTROLFLOW_",
        env_file=(
            ""
            if os.getenv("CONTROLFLOW_TEST_MODE")
            else ("~/.controlflow/.env", ".env")
        ),
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class PrefectSettings(ControlFlowSettings):
    """
    All settings here are used as defaults for Prefect, unless overridden by env vars.
    Note that `apply()` must be called before Prefect is imported.
    """

    PREFECT_LOGGING_LEVEL: str = "WARNING"
    PREFECT_EXPERIMENTAL_ENABLE_NEW_ENGINE: str = "true"

    def apply(self):
        import os

        if "prefect" in sys.modules:
            warnings.warn(
                "Prefect has already been imported; ControlFlow defaults will not be applied."
            )

        for k, v in self.model_dump().items():
            if k not in os.environ:
                os.environ[k] = v


class Settings(ControlFlowSettings):
    assistant_model: str = "gpt-4o"
    max_task_iterations: Union[int, None] = Field(
        None,
        description="The maximum number of iterations to attempt to complete a task "
        "before raising an error. If None, the task will run indefinitely. "
        "This setting can be overridden by the `max_iterations` attribute "
        "on a task.",
    )
    prefect: PrefectSettings = Field(default_factory=PrefectSettings)
    openai_api_key: Optional[str] = Field(None, validate_assignment=True)
    eager_mode: bool = Field(
        True,
        description="If True, @task- and @flow-decorated functions are run immediately. "
        "This can be set on a per-task or per-flow basis using the `eager` argument.",
    )

    # --- sending messages to Agents

    enable_local_input: bool = Field(
        True,
        description="If True, the user can provide input via "
        "the terminal. Otherwise, only API input is accepted.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.prefect.apply()

    @field_validator("openai_api_key", mode="after")
    def _apply_api_key(cls, v):
        if v is not None:
            import marvin

            marvin.settings.openai.api_key = v
        return v


settings = Settings()


@contextmanager
def temporary_settings(**kwargs: Any):
    """
    Temporarily override ControlFlow setting values, including nested settings objects.

    To override nested settings, use `__` to separate nested attribute names.

    Args:
        **kwargs: The settings to override, including nested settings.

    Example:
        Temporarily override log level and OpenAI API key:
        ```python
        import controlflow
        from controlflow.settings import temporary_settings

        # Override top-level settings
        with temporary_settings(log_level="INFO"):
            assert controlflow.settings.log_level == "INFO"
        assert controlflow.settings.log_level == "DEBUG"

        # Override nested settings
        with temporary_settings(openai__api_key="new-api-key"):
            assert controlflow.settings.openai.api_key.get_secret_value() == "new-api-key"
        assert controlflow.settings.openai.api_key.get_secret_value().startswith("sk-")
        ```
    """
    old_env = os.environ.copy()
    old_settings = deepcopy(settings)

    def set_nested_attr(obj: object, attr_path: str, value: Any):
        parts = attr_path.split("__")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    try:
        for attr_path, value in kwargs.items():
            set_nested_attr(settings, attr_path, value)
        yield

    finally:
        os.environ.clear()
        os.environ.update(old_env)

        for attr, value in old_settings:
            set_nested_attr(settings, attr, value)
