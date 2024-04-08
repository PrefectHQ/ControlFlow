import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlFlowSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_prefix="CONTROLFLOW_",
        env_file=(
            ""
            if os.getenv("CONTROLFLOW_TEST_MODE")
            else ("~/.control_flow/.env", ".env")
        ),
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class PrefectSettings(ControlFlowSettings):
    """
    All settings here are used as defaults for Prefect, unless overridden by env vars.
    """

    PREFECT_LOGGING_LEVEL: str = "WARNING"

    def apply(self):
        import os

        for k, v in self.model_dump().items():
            if k not in os.environ:
                os.environ[k] = v


class Settings(ControlFlowSettings):
    assistant_model: str = "gpt-4-1106-preview"
    max_agent_iterations: int = 10
    prefect: PrefectSettings = Field(default_factory=PrefectSettings)

    def __init__(self, **data):
        super().__init__(**data)
        self.prefect.apply()


settings = Settings()
