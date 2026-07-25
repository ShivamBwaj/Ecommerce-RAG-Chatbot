from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from api.core.config import config
from api.core.embeddings import get_embedding


def retrieve_data(query: str, qdrant_client: QdrantClient, top_k: int = 5) -> dict:
    query_embedding = get_embedding(query)

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


def get_item_payload_by_parent_asin(
    qdrant_client: QdrantClient,
    parent_asin: str,
) -> dict | None:
    records, _ = qdrant_client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="parent_asin",
                    match=MatchValue(value=parent_asin),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not records:
        return None

    return records[0].payload
