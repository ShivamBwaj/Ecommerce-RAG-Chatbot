from pydantic import BaseModel, Field
from typing import Optional



class FeedbackRequest(BaseModel):
    trace_id: str=Field(..., description="The trace ID for the request.")
    feedback_score: Optional[int]=Field(None, description="The feedback score.")
    feedback_text: str=Field(..., description="The feedback text.")
    thread_id: str=Field(..., description="The thread ID for the request.")
    feedback_source_type: str=Field(..., description="The type of the feedback source.")

class FeedbackResponse(BaseModel):
    request_id: str=Field(..., description="The request ID.")
    status: str=Field(..., description="The status of the feedback submission.")

class RAGRequest(BaseModel):
    query: str=Field(..., description="The user's query for retrieval-augmented generation.")
    thread_id: str=Field(..., description="The thread ID")

class RAGUsedContext(BaseModel):
    image_url: str=Field(..., description="The URL of the image used to answer the question")
    price: Optional[float]=Field(..., description="The price of the item used to answer the question")
    description: str=Field(..., description="The description of the item used to answer the question")

class RAGResponse(BaseModel):
    request_id: str=Field(..., description="The request ID.")
    answer: str=Field(..., description="The retrieved information for augmentation.")
    used_context: list[RAGUsedContext]=Field(..., description="The used context.")
    trace_id: str=Field(..., description="The trace ID for the request.")