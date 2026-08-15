
from api.agents.tools import retrieve_data
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
    MatchAny
)

from api.core.config import config
from api.core.embeddings import get_embedding



def process_items_context(context):
    formatted_context=""

    for id,chunk,rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context+=f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context



#### item description retrieval functions

def retrieve_items_data(query: str, top_k: int = 5) -> dict:
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    search_result = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            Prefetch(
                query=query_embedding,
                using=config.qdrant_dense_vector_name,
                limit=20,
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25",
                ),
                using=config.QDRANT_SPARSE_VECTOR_NAME,
                limit=20,
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    for point in search_result.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["description"])
        retrieved_context_ratings.append(point.payload["average_rating"])
        similarity_scores.append(point.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }
