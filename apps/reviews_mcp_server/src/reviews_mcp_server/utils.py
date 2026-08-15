
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


#### item reviews retrieval functions

def retrieve_reviews_data(query,item_list, top_k: int = 5) -> dict:
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    results=qdrant_client.query_points(
            collection_name="amazon-items-collection-02-openai-small-reviews",
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="parent_asin",
                                match=MatchAny(
                                    any=item_list
                                )
                            )
                        ]
                    ),
                    limit=20
                )
            ],
            query=FusionQuery(fusion="rrf"),
            limit=top_k
        )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    
    for point in results.points:
        retrieved_context_ids.append(point.payload["parent_asin"])
        retrieved_context.append(point.payload["text"])
        
        similarity_scores.append(point.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


def process_reviews_context(context):
    formatted_context=""

    for id,chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context+=f"- ID: {id}, review: {chunk}\n"
    return formatted_context

