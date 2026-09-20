from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):

    API_URL: str = "http://api:8000"

    # Ignore extra keys from the shared .env file (OPENAI_API_KEY, LANGSMITH_*, etc.)
    # so it doesn't fail validation just because it wasn't written for this app alone.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

config = Config()