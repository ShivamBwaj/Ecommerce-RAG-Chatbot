from openai import OpenAI
import instructor

from api.core.config import config


LLM_PROVIDER = config.llm_provider
LLM_MODEL = config.llm_model


def create_llm_client():
    if config.OPENAI_API_KEY:
        return instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    if config.GROQ_API_KEY:
        return instructor.from_provider(f"groq/{config.GROQ_MODEL}")

    raise RuntimeError("Set OPENAI_API_KEY or GROQ_API_KEY before starting the API.")