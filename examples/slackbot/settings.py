from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from raggy.vectorstores.chroma import ChromaClientType


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    slack_api_token: SecretStr
    chroma_client_type: ChromaClientType = "cloud"
    google_api_key: SecretStr
    google_cse_id: SecretStr


settings = Settings()  # type: ignore
