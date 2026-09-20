from fastapi import FastAPI
from api.core.config import config
from api.api.middleware import RequestIDMiddleware
import logging
from fastapi.middleware.cors import CORSMiddleware
from api.api.endpoints import api_router
from langgraph.checkpoint.postgres import PostgresSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)

app=FastAPI()
app.add_middleware(RequestIDMiddleware)


@app.on_event("startup")
def setup_checkpoint_tables():
    """Idempotent: creates the LangGraph checkpoint tables on first boot against
    whatever Postgres this environment points at, so a fresh database (a new
    Render deploy, a teammate's local docker-compose) never needs a manual
    migration step before the graph can run."""
    try:
        with PostgresSaver.from_conn_string(config.POSTGRES_DSN) as checkpointer:
            checkpointer.setup()
        logger.info("LangGraph checkpoint tables ready")
    except Exception:
        logger.exception("Failed to set up LangGraph checkpoint tables")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(api_router)

