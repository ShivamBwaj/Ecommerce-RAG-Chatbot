from pydantic import BaseModel, Field
from langchain_core.messages import convert_to_openai_messages
from langsmith import traceable

from typing import List

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.utils.utils import format_ai_message

import instructor



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

@traceable(name="agent node",run_type="llm",metadata={"ls_provider":"groq","ls_model_name":"groq/llama-3.3-70b-versatile"})
def agent_node(state)->dict:
    
    template=prompt_template_config("api/agents/prompts/qa_agent.yaml", "qa_agent")

    prompt=template.render(available_tools=state.available_tools)
    messages=state.messages
    conversation=[]
    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_provider("groq/llama-3.3-70b-versatile")

    response, raw_response = client.create_with_completion(
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt},*conversation],
        temperature=0.5,
    )
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

@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "groq", "ls_model_name": "llama-3.3-70b-versatile"}
)
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
    
    
    client = instructor.from_provider("groq/llama-3.3-70b-versatile")
    
    response, raw_response = client.create_with_completion(
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt},*conversation],
        temperature=0.5,
    )
    
    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }