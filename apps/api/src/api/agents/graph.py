
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition,MatchValue
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from operator import add
from api.agents.tools import get_formatted_context
from api.agents.utils.utils import get_tool_descriptions
import numpy as np
from typing import List, Dict, Any, Annotated
from api.agents.agents import ToolCall, RAGUsedContext , agent_node, intent_router_node




class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []

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

tools=[get_formatted_context]
tool_node=ToolNode(tools)
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

graph=workflow.compile()

#### agent execution function

def run_agent(question:str)->dict:

    initial_state={
        "messages":[{"role":"user","content":question}],
        "iteration":0,
        "available_tools":tool_descriptions
    }
    result =graph.invoke(initial_state)
    return result



def rag_agent_wrapper(question):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result= run_agent(question)

    used_context=[]
    dummy_vector=np.zeros(384).tolist()
    for item in result.get("references",[]):
        payload=qdrant_client.query_points(
            collection_name="amazon-items-collection-01-hybrid-search",
            query_vector=dummy_vector,
            limit=1,
            using="all-MiniLM-L6-v2",
            with_payload=True,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id)
                    )
                ]
            )
        ).points[0].payload
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
        "used_context": used_context
    }
