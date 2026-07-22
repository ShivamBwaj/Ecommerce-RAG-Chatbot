from qdrant_client import QdrantClient
from qdrant_client.models import Document, Filter, FieldCondition, FusionQuery, MatchValue, Prefetch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
from langsmith import traceable,get_current_run_tree
from pydantic import BaseModel, Field

from api.agents.utils.prompt_management import prompt_template_config, prompt_template_registry
import numpy as np
from api.core.llm import LLM_MODEL, LLM_PROVIDER, create_llm_client

class RAGUsedContext(BaseModel):
    id: str=Field(..., description="The ID of the item used to answer the question")
    description: str=Field(..., description="Short description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str=Field(..., description="The answer to the question")
    references: list[RAGUsedContext]=Field(..., description="List of Items used to answer the question")



from api.core.config import config


def _mean_pool_embedding(raw_embedding):
    if not raw_embedding:
        raise ValueError("Hugging Face API returned an empty embedding.")

    if isinstance(raw_embedding[0], (int, float)):
        return [float(value) for value in raw_embedding]

    if isinstance(raw_embedding[0], list):
        token_count = len(raw_embedding)
        vector_size = len(raw_embedding[0])
        pooled = [0.0] * vector_size

        for token_vector in raw_embedding:
            if len(token_vector) != vector_size:
                raise ValueError("Inconsistent token vector dimensions in HF embedding response.")
            for idx, value in enumerate(token_vector):
                pooled[idx] += float(value)

        return [value / token_count for value in pooled]

    raise ValueError("Unexpected Hugging Face embedding response format.")

@traceable(name="embed query",run_type="embedding",metadata={"ls_provider":"hugging-face","ls_model_name":"sentence-transformers/all-MiniLM-L6-v2"})
def get_embedding(text, model_name: str | None = None):
    selected_model = model_name or config.HF_EMBEDDING_MODEL
    endpoint = (
        "https://router.huggingface.co/hf-inference/models/"
        f"{selected_model}/pipeline/feature-extraction"
    )
    payload = json.dumps(
        {
            "inputs": text,
            "normalize": True,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if config.HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {config.HF_API_TOKEN}"

    request = Request(endpoint, data=payload, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Hugging Face embedding API request failed ({exc.code}): {message}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Hugging Face embedding API: {exc}") from exc

    if isinstance(response_data, dict) and response_data.get("error"):
        raise RuntimeError(f"Hugging Face embedding API error: {response_data['error']}")

    if isinstance(response_data, list) and len(response_data) == 1:
        return _mean_pool_embedding(response_data[0])
    

    return _mean_pool_embedding(response_data)

@traceable(name="retrieve data",run_type="retriever")
def retrieve_data(query,qdrant_client, top_k=5):

    query_embedding=get_embedding(query)

    search_result=qdrant_client.query_points(
        collection_name="amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="all-MiniLM-L6-v2",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25"
                ),
                using="bm25",
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
    )

    retrieved_context_ids=[]
    retrieved_context=[]
    similarity_scores=[]
    retrieved_context_ratings=[]
    for search_result in search_result.points:
        retrieved_context_ids.append(search_result.payload["parent_asin"])
        retrieved_context.append(search_result.payload["description"])
        retrieved_context_ratings.append(search_result.payload["average_rating"])
        similarity_scores.append(search_result.score)


    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }

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

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result= rag_pipeline(question,qdrant_client,top_k)

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
        "answer": result["answer"],
        "used_context": used_context
    }
