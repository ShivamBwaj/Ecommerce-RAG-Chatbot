from fastmcp import FastMCP
from reviews_mcp_server.utils import retrieve_reviews_data, process_reviews_context

mcp=FastMCP("reviews_mcp_server")

@mcp.tool()
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

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)