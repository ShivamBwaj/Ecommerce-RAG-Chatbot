
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from operator import add
from api.agents.tools import get_formatted_items_context,get_item_payload_by_parent_asin,get_formatted_reviews_context
from api.agents.utils.utils import get_tool_descriptions
from typing import List, Dict, Any, Annotated
from api.agents.agents import ToolCall, RAGUsedContext , agent_node, intent_router_node
from langgraph.checkpoint.postgres import PostgresSaver
from api.core.config import config as app_config
import json


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: List[RAGUsedContext] = []
    trace_id: str = ""

#### Edges

def tool_router(state:State)->str:
    "Decide weatehr to continue or end"
    # Pending tool calls win over final_answer: ending here would drop the call
    # and surface the model's pre-retrieval preamble as the final answer.
    if len(state.tool_calls)>0 and state.iteration <= 2:
        return "tools"
    return "end"
    

def intent_router_conditional_edges(state: State):
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

#### workflow
workflow = StateGraph(State)

tools=[get_formatted_items_context,get_formatted_reviews_context]


def handle_tool_error(error: Exception) -> str:
    """Return a ToolMessage when a retrieval tool fails.

    This lets the graph continue with a valid response for the tool call instead
    of leaving an unanswered tool call in the persisted conversation state.
    """
    return (
        "The retrieval tool could not complete this request. "
        f"Error: {error!s}. Please answer using the available results, "
        "or explain that no review information is currently available."
    )


tool_node=ToolNode(tools, handle_tool_errors=handle_tool_error)
tool_descriptions=get_tool_descriptions(tools)

workflow.add_node("agent_node",agent_node)
workflow.add_node("tool_node",tool_node)
workflow.add_node("intent_router_node", intent_router_node)
workflow.add_edge(START,"intent_router_node")


workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)
workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END
    }
)


workflow.add_edge("tool_node","agent_node")



def rag_agent_stream_wrapper(question:str,thread_id:str):
    

    def _string_for_sse(message:str):       ##server sent events
        return f"data: {message}\n\n"

    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _is_node_end(chunk):
            return chunk[0] == "updates"

        def _tool_to_text(tool_call):
            if tool_call.name == "get_formatted_items_context":
                return f"Looking for items: {tool_call.arguments.get('query', '')}."
            elif tool_call.name == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            else:
                return f"Unknown tool: {tool_call.name}."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).tool_calls])
                return message
        else:
            return False

    qdrant_client = QdrantClient(url=app_config.QDRANT_URL)
    initial_state={
            "messages":[{"role":"user","content":question}],
            "iteration":0,
            "available_tools":tool_descriptions
    }
    conffig={
            "configurable":{    
                "thread_id":thread_id
            }
    }
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
            graph=workflow.compile(checkpointer=checkpointer)
            for chunk in graph.stream(initial_state,
                                        config=conffig,
                                        stream_mode=["updates","debug","values"]):
                processed_chunk=_process_graph_event(chunk)

                if processed_chunk:
                    yield _string_for_sse(processed_chunk)
                if chunk[0]=="values":
                    result=chunk[1]

    used_context=[]
    for item in result.get("references",[]):
        payload=get_item_payload_by_parent_asin(qdrant_client, item.id)
        if not payload:
            continue
        image_url=payload.get("image")
        price=payload.get("price")
        if image_url:
            used_context.append({
                "image_url": image_url,
                "price": price,
                "description": item.description
            })

    yield _string_for_sse(json.dumps(
        {
            "type": "final_result",
            "data":{
                "answer": result.get("answer", ""),
                "used_context": used_context,
                "trace_id": result.get("trace_id", "")
            }
        }
    ))