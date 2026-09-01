from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    HF_API_TOKEN: str | None = None
    EMBEDDING_PROVIDER: str = "openai"
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HF_DENSE_VECTOR_NAME: str = "all-MiniLM-L6-v2"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536
    OPENAI_MODEL: str = "gpt-5.4"
    GROQ_MODEL: str = "qwen/qwen3-32b"
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "amazon-items-collection-02-openai-small"
    QDRANT_SPARSE_VECTOR_NAME: str = "bm25"

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

    @property
    def embedding_provider(self) -> str:
        return self.EMBEDDING_PROVIDER.strip().lower()

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider == "openai":
            return self.OPENAI_EMBEDDING_MODEL
        if self.embedding_provider in {"huggingface", "hugging-face", "hf"}:
            return self.HF_EMBEDDING_MODEL
        raise ValueError(
            "EMBEDDING_PROVIDER must be one of: openai, huggingface."
        )

    @property
    def qdrant_dense_vector_name(self) -> str:
        if self.embedding_provider == "openai":
            return self.OPENAI_EMBEDDING_MODEL
        return self.HF_DENSE_VECTOR_NAME

    @property
    def embedding_dimensions(self) -> int:
        if self.embedding_provider == "openai":
            return self.OPENAI_EMBEDDING_DIMENSIONS
        return 384


config = Config()
