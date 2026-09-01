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
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from qdrant_client.models import MatchValue


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
    qdrant_client = QdrantClient(url="http://localhost:6333")
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
    qdrant_client = QdrantClient(url="http://localhost:6333")
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

#### add to shopping cart tools

def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:
    """Add a list of provided items to the shopping cart.
    
    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.
    
    Returns:
        A list of the items added to the shopping cart.
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password"
    )
    conn.autocommit = True
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        for item in items:
            product_id = item['product_id']
            quantity = item['quantity']
            
            qdrant_client = QdrantClient(url="http://localhost:6333")
            
            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
                collection_name="amazon-items-collection-02-openai-small",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id)
                                )
                            ]
                        ),
                        using="text-embedding-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            ).points[0].payload
            
            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = 'USD'
            
            # Check if item already exists
            check_query = """
                SELECT id, quantity, price
                FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()
            
            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity
                
                update_query = """
                    UPDATE shopping_carts.shopping_cart_items
                    SET
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """
                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))
            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """
                cursor.execute(insert_query, (user_id, cart_id, product_id, price, quantity, currency, product_image_url))
    
    return f"Added {items} to the shopping cart."



def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:
    """
    Retrieve all items in a user's shopping cart.
    
    Args:
        user_id: User ID
        cart_id: Cart identifier
    
    Returns:
        List of dictionaries containing cart items
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password"
    )
    conn.autocommit = True
    
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
            SELECT
                product_id, price, quantity,
                currency, product_image_url,
                (price * quantity) as total_price
            FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s
            ORDER BY added_at DESC
        """
        cursor.execute(query, (user_id, cart_id))
        
        return [dict(row) for row in cursor.fetchall()]



def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:
    """
    Remove an item completely from the shopping cart.

    Args:
        user_id: User ID
        product_id: Product ID to remove
        cart_id: Cart identifier

    Returns:
        True if item was removed, False if item wasn't found
    """
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="langgraph_user",
        password="langgraph_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
            DELETE FROM shopping_carts.shopping_cart_items
            WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
        """
        cursor.execute(query, (user_id, cart_id, product_id))

        return cursor.rowcount > 0