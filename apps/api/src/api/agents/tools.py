from langsmith import traceable
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



@traceable(name="format retrieved context",run_type="prompt")
def process_items_context(context):
    formatted_context=""

    for id,chunk,rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context+=f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context


def get_formatted_items_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
    query: The query to get the top k context for
    top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
    A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """
    from qdrant_client import QdrantClient
    from api.core.config import config

    context = retrieve_items_data(query, top_k)  
    formatted_context = process_items_context(context)

    return formatted_context

#### item description retrieval functions
@traceable(name="retrieve data", run_type="retriever")
def retrieve_items_data(query: str, top_k: int = 5) -> dict:
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
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

#### item reviews retrieval functions

@traceable(name="retrieve reviews data", run_type="retriever")
def retrieve_reviews_data(query,item_list, top_k: int = 5) -> dict:
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient(url=config.QDRANT_URL)
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


@traceable(name="format retrieved reviews context",run_type="prompt")
def process_reviews_context(context):
    formatted_context=""

    for id,chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context+=f"- ID: {id}, review: {chunk}\n"
    return formatted_context

def get_formatted_reviews_context(query: str,item_list:list, top_k: int = 15) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.
    Args:
        query: The query to get the top k reviews for
        item_list: A list of prefiltered items to retrieve reviews for
        top_k: The number of reviews to retrieve, works best with 5 or more
    
    Returns:
        A string of the top k reviews with IDs prepending each review, each representing an inventory item for a given query.
    """

    context = retrieve_reviews_data(query, item_list, top_k=top_k)
    formatted_context = process_reviews_context(context)

    return formatted_context
