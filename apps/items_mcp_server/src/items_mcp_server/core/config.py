from pydantic import BaseSettings,SettingsConfigDict

class Config(BaseSettings):
    OPEN_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env")
    
config = Config()