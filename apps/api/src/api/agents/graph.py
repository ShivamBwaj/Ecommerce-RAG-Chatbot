
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from operator import add
from api.agents.tools import get_formatted_context,get_item_payload_by_parent_asin,get_formatted_reviews_context
from api.agents.utils.utils import get_tool_descriptions
from typing import List, Dict, Any, Annotated
from api.agents.agents import ToolCall, RAGUsedContext , agent_node, intent_router_node
from langgraph.checkpoint.postgres import PostgresSaver
from api.core.config import config as app_config



class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []
    trace_id: str = ""

#### Edges

def tool_router(state:State)->str:
    "Decide weatehr to continue or end"
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls)>0:
        return "tools"
    else:
        return "end"
    

def intent_router_conditional_edges(state: State):
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"

#### workflow
workflow = StateGraph(State)

tools=[get_formatted_context,get_formatted_reviews_context]


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



#### agent execution function

def run_agent(question:str,thread_id:str)->dict:

    initial_state={
        "messages":[{"role":"user","content":question}],
        "iteration":0,
        "available_tools":tool_descriptions
    }
    config={
        "configurable":{    
            "thread_id":thread_id
        }
    }
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph=workflow.compile(checkpointer=checkpointer)
        result=graph.invoke(initial_state,config=config)

    return result



def rag_agent_wrapper(question,thread_id):

    qdrant_client = QdrantClient(url=app_config.QDRANT_URL)

    result= run_agent(question, thread_id)

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

    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
        "trace_id": result.get("trace_id", "")
    }
