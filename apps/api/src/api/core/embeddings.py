import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langsmith import traceable
from openai import OpenAI

from api.core.config import config


def _mean_pool_embedding(raw_embedding):
    if not raw_embedding:
        raise ValueError("Hugging Face API returned an empty embedding.")

    if isinstance(raw_embedding[0], (int, float)):
        return [float(value) for value in raw_embedding]

    if isinstance(raw_embedding[0], list):
        token_count = len(raw_embedding)
        vector_size = len(raw_embedding[0])
        pooled = [0.0] * vector_size

        for token_vector in raw_embedding:
            if len(token_vector) != vector_size:
                raise ValueError("Inconsistent token vector dimensions in HF embedding response.")
            for idx, value in enumerate(token_vector):
                pooled[idx] += float(value)

        return [value / token_count for value in pooled]

    raise ValueError("Unexpected Hugging Face embedding response format.")


def _get_huggingface_embedding(text: str, model_name: str) -> list[float]:
    endpoint = (
        "https://router.huggingface.co/hf-inference/models/"
        f"{model_name}/pipeline/feature-extraction"
    )
    payload = json.dumps(
        {
            "inputs": text,
            "normalize": True,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if config.HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_API_TOKEN}"

    request = Request(endpoint, data=payload, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Hugging Face embedding API request failed ({exc.code}): {message}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Hugging Face embedding API: {exc}") from exc

    if isinstance(response_data, dict) and response_data.get("error"):
        raise RuntimeError(f"Hugging Face embedding API error: {response_data['error']}")

    if isinstance(response_data, list) and len(response_data) == 1:
        return _mean_pool_embedding(response_data[0])

    return _mean_pool_embedding(response_data)


def _get_openai_embedding(text: str, model_name: str) -> list[float]:
    return get_embeddings([text], model_name=model_name)[0]


def get_embeddings(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    selected_model = model_name or config.embedding_model

    if config.embedding_provider != "openai":
        return [_get_huggingface_embedding(text, selected_model) for text in texts]

    if not config.OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY before using OpenAI embeddings.")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(
        model=selected_model,
        input=texts,
        dimensions=config.OPENAI_EMBEDDING_DIMENSIONS,
    )
    embeddings_by_index = sorted(response.data, key=lambda embedding: embedding.index)
    return [
        [float(value) for value in embedding.embedding]
        for embedding in embeddings_by_index
    ]


@traceable(
    name="embed query",
    run_type="embedding",
    metadata={
        "ls_provider": config.embedding_provider,
        "ls_model_name": config.embedding_model,
    },
)
def get_embedding(text: str, model_name: str | None = None) -> list[float]:
    selected_model = model_name or config.embedding_model

    if config.embedding_provider == "openai":
        return _get_openai_embedding(text, selected_model)

    return _get_huggingface_embedding(text, selected_model)
