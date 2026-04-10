from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    GROQ_API_KEY: str
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    HF_API_TOKEN: str | None = None
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    QDRANT_URL: str = "http://qdrant:6333"

    # Ignore extra keys from .env (e.g. LANGSMITH_*, HUGGINGFACEHUB_API_TOKEN) so shared env files do not fail validation.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def gemini_api_key(self) -> str:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or ""


config = Config()
