import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, ValidationError

load_dotenv("app/.env")

class Settings(BaseSettings):
    """Settings for the application."""
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
    EMBEDDING_SIZE: int = os.getenv("EMBEDDING_SIZE", "768")   # Assuming the embedding size is 3, adjust as necessary
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    PG_VECTOR_DB_NAME: str = os.getenv("PG_VECTOR_DB_NAME", "vector_db")
    PG_VECTOR_DB_USERNAME: str = os.getenv("PG_VECTOR_DB_USERNAME", "postgres")
    PG_VECTOR_DB_PASSWORD: str = os.getenv("PG_VECTOR_DB_PASSWORD", "12345")
    PG_VECTOR_DB_HOST: str = os.getenv("PG_VECTOR_DB_HOST", "localhost")
    PG_VECTOR_DB_PORT: int = int(os.getenv("PG_VECTOR_DB_PORT", "5432"))
    PG_VECTOR_DB_VECTOR_SIZE: int = int(os.getenv("PG_VECTOR_DB_VECTOR_SIZE", "768"))  # Adjust as necessary
    CHAT_DB_NAME: str = os.getenv("CHAT_DB_NAME", "chat_db")
    CHAT_DB_USERNAME: str = os.getenv("CHAT_DB_USERNAME", "postgres")
    CHAT_DB_PASSWORD: str = os.getenv("CHAT_DB_PASSWORD", "12345")
    CHAT_DB_HOST: str = os.getenv("CHAT_DB_HOST", "localhost")
    CHAT_DB_PORT: int = int(os.getenv("CHAT_DB_PORT", "5432"))

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.GEMINI_API_KEY:
            raise ValidationError("GEMINI_API_KEY is missing. Please set it in the environment variables.")

try:
    config_settings = Settings()
except ValidationError as e:
    print(f"Error: {e}")
    raise SystemExit(1)