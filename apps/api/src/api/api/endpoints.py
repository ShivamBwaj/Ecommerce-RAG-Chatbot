from fastapi import APIRouter,Request

from api.api.models import RAGRequest, RAGResponse,RAGUsedContext
import logging

from api.agents.graph import rag_agent_wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)



rag_router = APIRouter()


@rag_router.post("/")
def rag(
    request: Request,
    payload: RAGRequest
) -> RAGResponse:
    answer=rag_agent_wrapper(payload.query)

    return RAGResponse(
        request_id=request.state.request_id,
        answer=answer["answer"],
        used_context=[RAGUsedContext(**used_context) for used_context in answer["used_context"]]
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"]) 