from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env values are populated in os.environ for external SDKs like LangSmith
load_dotenv()


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "AI-ASSIGNMENT"
    ENV: str = "development"
    API_PREFIX: str = "/api"
    VERSION: str = "0.0.1"

    GEMINI_BASE_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    GEMINI_FREE_API_KEY: str | None = None

    GEMINI_PAID_API_KEY: str | None = None

    GEMINI_THINKING_MODEL: str | None = "gemini-3.8-flash"

    GEMINI_MODEL: str = "gemini-3.1-flash-lite"



    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    OPENROUTER_API_KEY: str | None = None

    OPENROUTER_FREE_MODEL: str | None = "openrouter/free"

    OPENROUTER_PAID_MODEL: str | None = "deepseek/deepseek-v4.1-flash"

    OPENROUTER_THINKING_MODEL: str | None = "deepseek/deepseek-v4-pro-0813"

    OPENROUTER_RERANK_MODEL: str = (
        "voyageai/rerank-2.5-lite"
    )

    OPENROUTER_SEARCH_MODEL: str | None = (
        "perplexity/sonar"
    )


    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    GROQ_API_KEY: str | None = None

    GROQ_MODEL: str = "openai/gpt-oss-120b"

    OCR_API_KEY: str | None = None


settings = Settings()