
from qdrant_client.models import Document,FusionQuery, Prefetch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from langsmith import traceable


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

@traceable(name="embed query",run_type="embedding",metadata={"ls_provider":"hugging-face","ls_model_name":"sentence-transformers/all-MiniLM-L6-v2"})
def get_embedding(text, model_name: str | None = None):
    selected_model = model_name or config.HF_EMBEDDING_MODEL
    endpoint = (
        "https://router.huggingface.co/hf-inference/models/"
        f"{selected_model}/pipeline/feature-extraction"
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

@traceable(name="retrieve data",run_type="retriever")
def retrieve_data(query,qdrant_client, top_k=5):

    query_embedding=get_embedding(query)

    search_result=qdrant_client.query_points(
        collection_name="amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="all-MiniLM-L6-v2",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
    )

    retrieved_context_ids=[]
    retrieved_context=[]
    similarity_scores=[]
    retrieved_context_ratings=[]
    for search_result in search_result.points:
        retrieved_context_ids.append(search_result.payload["parent_asin"])
        retrieved_context.append(search_result.payload["description"])
        retrieved_context_ratings.append(search_result.payload["average_rating"])
        similarity_scores.append(search_result.score)


    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }

@traceable(name="format retrieved context",run_type="prompt")
def process_context(context):
    formatted_context=""

    for id,chunk,rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context+=f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context



def get_formatted_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
    query: The query to get the top k context for
    top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
    A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    context = retrieve_data(query, qdrant_client, top_k)  
    formatted_context = process_context(context)

    return formatted_context

