from langsmith import traceable

from api.agents.retrieval import retrieve_data

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
    from api.core.config import config

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    context = retrieve_data(query, qdrant_client, top_k)  
    formatted_context = process_context(context)

    return formatted_context

