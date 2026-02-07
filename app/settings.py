"""
Application settings with pydantic BaseSettings for environment variable handling.
Fails fast if required environment variables are missing.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Required - will fail fast if not provided
    one_min_ai_api_key: str = Field(
        ...,
        description="1minAI API key (required)",
        alias="ONE_MIN_AI_API_KEY",
    )

    # Optional with defaults
    one_min_ai_base_url: str = Field(
        default="https://api.1min.ai",
        description="1minAI base URL",
        alias="ONE_MIN_AI_BASE_URL",
    )
    proxy_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the proxy server",
        alias="PROXY_HOST",
    )
    proxy_port: int = Field(
        default=8080,
        description="Port for the proxy server",
        alias="PROXY_PORT",
    )
    default_1min_model: str = Field(
        default="gpt-4o-mini",
        description="Default model to use when not specified",
        alias="DEFAULT_1MIN_MODEL",
    )
    default_websearch: bool = Field(
        default=False,
        description="Enable web search by default",
        alias="DEFAULT_WEBSEARCH",
    )
    default_num_of_site: int = Field(
        default=1,
        description="Number of sites for web search",
        alias="DEFAULT_NUM_OF_SITE",
    )
    default_max_word: int = Field(
        default=500,
        description="Maximum words from web search",
        alias="DEFAULT_MAX_WORD",
    )
    request_timeout_secs: int = Field(
        default=60,
        description="Request timeout in seconds",
        alias="REQUEST_TIMEOUT_SECS",
    )
    retries: int = Field(
        default=2,
        description="Number of retries for failed requests",
        alias="RETRIES",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
        alias="LOG_LEVEL",
    )
    enable_request_logging: bool = Field(
        default=False,
        description="Enable request logging (redacted)",
        alias="ENABLE_REQUEST_LOGGING",
    )
    extra_models: Optional[str] = Field(
        default=None,
        description="Comma-separated list of extra models for /v1/models",
        alias="EXTRA_MODELS",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v

    def get_extra_models_list(self) -> list[str]:
        """Parse extra_models into a list of model names."""
        if not self.extra_models:
            return []
        return [m.strip() for m in self.extra_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance. Fails fast if required env vars missing."""
    return Settings()
