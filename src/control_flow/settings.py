import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class ControlFlowSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            ""
            if os.getenv("CONTROL_FLOW_TEST_MODE")
            else ("~/.control_flow/.env", ".env")
        ),
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class Settings(ControlFlowSettings):
    assistant_model: str = "gpt-4-1106-preview"


settings = Settings()
