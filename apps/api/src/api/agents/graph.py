
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from operator import add
from api.agents.tools import (
    get_formatted_items_context,
    get_item_payload_by_parent_asin,
    get_formatted_reviews_context,
    add_to_shopping_cart,
    get_shopping_cart,
    remove_from_cart,
)
from api.agents.utils.utils import get_tool_descriptions
from typing import List, Dict, Any, Annotated
from api.agents.agents import (
    ToolCall,
    RAGUsedContext,
    AgentProperties,
    CoordinatorAgentProperties,
    product_qa_agent,
    shopping_cart_agent,
    coordinator_agent,
)
from langgraph.checkpoint.postgres import PostgresSaver
from api.core.config import config as app_config
import json
import logging
import time

# Free-tier LLM keys (e.g. Groq) rate-limit and back off for several seconds per
# call; with multiple agent hops per turn that can compound past what a client
# or proxy will wait on. Bail out with a clear message instead of hanging.
STREAM_TIMEOUT_SECONDS = 55


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    trace_id: str = ""
    user_id: str = ""
    cart_id: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(default_factory=CoordinatorAgentProperties)


#### Edges

def product_qa_agent_tool_router(state: State) -> str:
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def shopping_cart_agent_tool_router(state: State) -> str:
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def coordinator_agent_edge(state: State) -> str:
    if state.coordinator_agent.iteration > 3:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    else:
        return "end"


#### workflow
workflow = StateGraph(State)

product_qa_agent_tools = [get_formatted_items_context, get_formatted_reviews_context]
shopping_cart_agent_tools = [add_to_shopping_cart, get_shopping_cart, remove_from_cart]


def handle_tool_error(error: Exception) -> str:
    """Return a ToolMessage when a retrieval/cart tool fails.

    This lets the graph continue with a valid response for the tool call instead
    of leaving an unanswered tool call in the persisted conversation state.
    The message is deliberately generic - the underlying error (HTTP status,
    stack trace, connection string, etc.) is logged server-side, not handed to
    the LLM, since it otherwise ends up quoted verbatim in the user-facing answer.
    """
    logging.getLogger(__name__).warning("Tool call failed: %s", error)
    return (
        "The tool could not complete this request due to a temporary issue. "
        "Please answer using the available results, or explain that the action "
        "could not be completed and ask the user to try again shortly."
    )


product_qa_agent_tool_node = ToolNode(product_qa_agent_tools, handle_tool_errors=handle_tool_error)
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools, handle_tool_errors=handle_tool_error)
product_qa_agent_tool_descriptions = get_tool_descriptions(product_qa_agent_tools)
shopping_cart_agent_tool_descriptions = get_tool_descriptions(shopping_cart_agent_tools)

workflow.add_node("coordinator_agent", coordinator_agent)
workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("product_qa_agent_tool_node", product_qa_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END,
    },
)
workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agent_tool_router,
    {
        "tools": "product_qa_agent_tool_node",
        "end": "coordinator_agent",
    },
)
workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_agent_tool_node",
        "end": "coordinator_agent",
    },
)

workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")


def rag_agent_stream_wrapper(question: str, thread_id: str, user_id: str = "", cart_id: str = ""):

    def _string_for_sse(message: str):  ##server sent events
        return f"data: {message}\n\n"

    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _tool_to_text(tool_call):
            if tool_call.name == "get_formatted_items_context":
                return f"Looking for items: {tool_call.arguments.get('query', '')}."
            elif tool_call.name == "get_formatted_reviews_context":
                return "Fetching user reviews..."
            elif tool_call.name == "add_to_shopping_cart":
                return "Adding items to your cart..."
            elif tool_call.name == "get_shopping_cart":
                return "Checking your cart..."
            elif tool_call.name == "remove_from_cart":
                return "Removing item from your cart..."
            else:
                return f"Unknown tool: {tool_call.name}."

        if _is_node_start(chunk):
            node_name = chunk[1].get("payload", {}).get("name")
            if node_name == "coordinator_agent":
                return "Planning..."
            if node_name == "product_qa_agent":
                return "Looking into the products..."
            if node_name == "shopping_cart_agent":
                return "Managing your cart..."
            if node_name == "product_qa_agent_tool_node":
                agent_state = chunk[1].get("payload", {}).get("input", {}).product_qa_agent
                return " ".join(_tool_to_text(tc) for tc in agent_state.tool_calls)
            if node_name == "shopping_cart_agent_tool_node":
                agent_state = chunk[1].get("payload", {}).get("input", {}).shopping_cart_agent
                return " ".join(_tool_to_text(tc) for tc in agent_state.tool_calls)
        return False

    qdrant_client = QdrantClient(url=app_config.QDRANT_URL, api_key=app_config.QDRANT_API_KEY)
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "user_id": user_id,
        "cart_id": cart_id,
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_agent_tool_descriptions,
            "tool_calls": [],
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_agent_tool_descriptions,
            "tool_calls": [],
        },
    }
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    start_time = time.monotonic()
    result = None
    timed_out = False
    with PostgresSaver.from_conn_string(app_config.POSTGRES_DSN) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        for chunk in graph.stream(
            initial_state,
            config=config,
            stream_mode=["updates", "debug", "values"],
        ):
            if time.monotonic() - start_time > STREAM_TIMEOUT_SECONDS:
                timed_out = True
                break

            processed_chunk = _process_graph_event(chunk)

            if processed_chunk:
                yield _string_for_sse(processed_chunk)
            if chunk[0] == "values":
                result = chunk[1]

    if timed_out or result is None:
        yield _string_for_sse(json.dumps(
            {
                "type": "final_result",
                "data": {
                    "answer": "This is taking longer than usual, likely an LLM provider rate limit. Please try again in a moment.",
                    "used_context": [],
                    "trace_id": "",
                },
            }
        ))
        return

    if not result.get("answer"):
        # The coordinator hit its iteration cap (coordinator_agent_edge -> "end")
        # without ever setting final_answer=True, so "answer" was never populated.
        result["answer"] = (
            "I wasn't able to fully answer that within the usual number of steps. "
            "Could you rephrase your question, or ask for one thing at a time?"
        )

    used_context = []
    for item in result.get("references", []):
        payload = get_item_payload_by_parent_asin(qdrant_client, item.id)
        if not payload:
            continue
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.description,
            })

    yield _string_for_sse(json.dumps(
        {
            "type": "final_result",
            "data": {
                "answer": result.get("answer", ""),
                "used_context": used_context,
                "trace_id": result.get("trace_id", ""),
            },
        }
    ))
