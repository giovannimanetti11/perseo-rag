from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PERSEO_",
        extra="ignore",
    )

    environment: Environment = "development"
    database_url: str = "postgresql+psycopg://perseo_app:perseo_app@localhost:5432/perseo_rag"
    migration_database_url: str = "postgresql+psycopg://perseo:perseo@localhost:5432/perseo_rag"
    log_level: LogLevel = "INFO"
    expose_api_docs: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
