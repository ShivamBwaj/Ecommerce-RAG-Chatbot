from pydantic import BaseModel, Field
from langchain_core.messages import convert_to_openai_messages
from langsmith import traceable,get_current_run_tree

from typing import List

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message
from api.core.llm import LLM_MODEL, LLM_PROVIDER, create_llm_client



#### response models

class IntentRouterResponse(BaseModel):
   question_relevant: bool
   answer: str

class ToolCall(BaseModel):
    name:str
    arguments:dict

class RAGUsedContext(BaseModel):
    id: str=Field(..., description="The ID of the item used to answer the question")
    description: str=Field(..., description="Short description of the item used to answer the question")

class AgentResponse(BaseModel):
    answer: str=Field(..., description="The answer to the user's question")
    references: List[RAGUsedContext]=Field(..., description="The list of retrieved contexts used to answer the question, each representing an inventory item")
    final_answer: bool=False
    tool_calls: List[ToolCall]=[]




###QnA agent node

@traceable(name="agent node",run_type="llm",metadata={"ls_provider": LLM_PROVIDER, "ls_model_name": LLM_MODEL})
def agent_node(state)->dict:
    
    template=prompt_template_config("api/agents/prompts/qa_agent.yaml", "qa_agent")

    prompt=template.render(available_tools=state.available_tools)
    messages=state.messages
    conversation=[]
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = create_llm_client()

    response, raw_response = client.create_with_completion(
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt},*conversation],
        model=LLM_MODEL,
        temperature=0.5,
    )

    current_run=get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message=format_ai_message(response)
    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references,
    }


### intent router agent node

@traceable(name="intent_router_node", run_type="llm", metadata={"ls_provider": LLM_PROVIDER, "ls_model_name": LLM_MODEL})
def intent_router_node(state):
    """
    Routes user queries by determining if they are relevant to products in stock.
    
    Args:
        state: Contains the initial_query from the user
        
    Returns:
        Dictionary with question_relevant flag and optional answer
    """
    
    template=prompt_template_config("api/agents/prompts/intent_router_agent.yaml", "intent_router_agent")

    prompt = template.render()
    
    messages=state.messages
    conversation=[]
    for message in messages:
        conversation.append(convert_to_openai_messages(message))
    
    
    client = create_llm_client()
    
    response, raw_response = client.create_with_completion(
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt},*conversation],
        model=LLM_MODEL,
        temperature=0.5,
    )
    current_run=get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id=str(getattr(current_run, "trace_id",current_run.id))
    else:
        trace_id=None

    
    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id
    }