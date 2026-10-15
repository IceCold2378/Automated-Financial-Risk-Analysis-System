"""
Application configuration using pydantic-settings.
All config is loaded from environment variables / .env file.
"""

import os
from enum import Enum
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- General ---
    APP_NAME: str = "Intelligent Underwriting Assistant"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # --- Security ---
    SECRET_KEY: str = Field(
        default="change-me-in-production-use-a-real-secret-key",
        description="Secret key for JWT encoding"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:8501",
        "http://localhost:3000",
    ]

    # --- Database ---
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./underwriting.db",
        description="Database connection URL (PostgreSQL recommended for production)"
    )

    # --- LLM ---
    LLM_PROVIDER: LLMProvider = LLMProvider.OLLAMA
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"

    # --- Azure OpenAI ---
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"

    # --- Anthropic ---
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # --- OpenRouter ---
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "anthropic/claude-sonnet-4-20250514"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # --- Agent ---
    AGENT_MAX_ITERATIONS: int = 5
    AGENT_ENABLE_TOOLS: bool = True
    AGENT_VERBOSE: bool = False

    # --- Vector DB ---
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "underwriting_guidelines"
    VECTOR_SEARCH_K: int = 5
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "baai/bge-large-en-v1.5"  # OpenRouter/OpenAI embedding model

    # --- Document Processing ---
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: list[str] = ["pdf", "docx", "txt"]

    # --- Rate Limiting & Caching ---
    RATE_LIMIT_PER_MINUTE: int = 30
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- Integrations: Salesforce ---
    SF_USERNAME: str = ""
    SF_PASSWORD: str = ""
    SF_SECURITY_TOKEN: str = ""
    SF_DOMAIN: str = "login"  # "login" for production, "test" for sandbox

    # --- Integrations: Snowflake ---
    SNOWFLAKE_ACCOUNT: str = ""
    SNOWFLAKE_USER: str = ""
    SNOWFLAKE_PASSWORD: str = ""
    SNOWFLAKE_DATABASE: str = "UNDERWRITING"
    SNOWFLAKE_WAREHOUSE: str = "COMPUTE_WH"
    SNOWFLAKE_SCHEMA: str = "PUBLIC"

    # --- Integrations: Azure ---
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER: str = "underwriting-docs"
    AZURE_KEYVAULT_URL: str = ""
    AZURE_SERVICEBUS_CONNECTION_STRING: str = ""
    AZURE_SERVICEBUS_QUEUE: str = "underwriting-events"

    # --- Guidelines ---
    GUIDELINES_PATH: str = Field(
        default="",
        description="Path to guidelines file (.txt or page-level .jsonl; defaults to the HUD 4000.1 corpus)"
    )

    @property
    def effective_guidelines_path(self) -> str:
        if self.GUIDELINES_PATH:
            return self.GUIDELINES_PATH
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, "data", "hud_4000_1_borrower_underwriting.jsonl")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
