from qdrant_client import QdrantClient
from langsmith import traceable,get_current_run_tree
from pydantic import BaseModel, Field

from api.agents.tools import get_item_payload_by_parent_asin, retrieve_data
from api.agents.utils.prompt_management import prompt_template_config
from api.core.llm import LLM_MODEL, LLM_PROVIDER, create_llm_client
from api.core.config import config

class RAGUsedContext(BaseModel):
    id: str=Field(..., description="The ID of the item used to answer the question")
    description: str=Field(..., description="Short description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str=Field(..., description="The answer to the question")
    references: list[RAGUsedContext]=Field(..., description="List of Items used to answer the question")

@traceable(name="format retrieved context",run_type="prompt")
def process_context(context):
    formatted_context=""

    for id,chunk,rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context+=f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context


### create a prompt for the LLM using the retrieved context and the user query

@traceable(name="build prompt",run_type="prompt")
def build_prompt(preprocessed_context, question):
    template=prompt_template_config("api/agents/prompts/retrieval_generation.yaml", "retrieval_generation")
    prompt=template.render(preprocessed_context=preprocessed_context, question=question)
    return prompt
    

### Generate Answer function


@traceable(name="generate answer", run_type="llm", metadata={"ls_provider": LLM_PROVIDER, "ls_model_name": LLM_MODEL})
def generate_answer(prompt):
    """
    Generate answer using Groq LLM.
    
    Model: configured via OPENAI_MODEL or GROQ_MODEL
    
    Args:
        prompt (str): The formatted prompt with context and question
        
    Returns:
        str: Generated answer from the LLM
    """
    client = create_llm_client()
    completion,raw_response = client.create_with_completion(
        messages=[
        {
            "role": "system",
            "content": prompt
        }
        ],
        model=LLM_MODEL,
        temperature=0,
        response_model=RAGGenerationResponse,
    )
    

    current_run=get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"]={
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
            "resoning_tokens": getattr(getattr(raw_response.usage, "completion_tokens_details", None), "reasoning_tokens", 0)

        }
    return completion

@traceable(
    name="RAG pipeline"
)
def rag_pipeline(query,qdrant_client,top_k=5):
    

    retrieved_context=retrieve_data(query,qdrant_client, top_k)
    preprocessed_context=process_context(retrieved_context)
    prompt=build_prompt(preprocessed_context, query)
    answer=generate_answer(prompt)

    final_result = {
        "answer": answer.answer,
        "references": answer.references,
        "question": query,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"],
    }

    return final_result


def rag_pipeline_wrapper(question,top_k=5):

    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result= rag_pipeline(question,qdrant_client,top_k)

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
        "answer": result["answer"],
        "used_context": used_context
    }
