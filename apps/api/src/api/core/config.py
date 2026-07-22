from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    HF_API_TOKEN: str | None = None
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    OPENAI_MODEL: str = "gpt-5.4"
    GROQ_MODEL: str = "qwen/qwen3-32b"
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

    @property
    def llm_provider(self) -> str:
        if self.OPENAI_API_KEY:
            return "openai"
        if self.GROQ_API_KEY:
            return "groq"
        return "none"

    @property
    def llm_model(self) -> str:
        if self.OPENAI_API_KEY:
            return self.OPENAI_MODEL
        return self.GROQ_MODEL


config = Config()
