import itertools

from openai import OpenAI
from groq import Groq
import instructor

from api.core.config import config


LLM_PROVIDER = config.llm_provider
LLM_MODEL = config.llm_model

_groq_api_keys = [
    key
    for key in (
        config.GROQ_API_KEY,
        config.GROQ_API_KEY2,
        config.GROQ_API_KEY3,
        config.GROQ_API_KEY4,
        config.GROQ_API_KEY5,
        config.GROQ_API_KEY6,
        config.GROQ_API_KEY7,
    )
    if key
]
# Round-robins across keys per call so a single free-tier Groq key's
# tokens-per-minute limit doesn't bottleneck every agent hop in a request.
_groq_key_cycle = itertools.cycle(_groq_api_keys) if _groq_api_keys else None


def create_llm_client():
    if config.OPENAI_API_KEY:
        return instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))

    if _groq_key_cycle:
        return instructor.from_groq(Groq(api_key=next(_groq_key_cycle)))

    raise RuntimeError("Set OPENAI_API_KEY or GROQ_API_KEY before starting the API.")