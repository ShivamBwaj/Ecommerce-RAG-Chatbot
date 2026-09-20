import asyncio
import math
import os
import time

from dotenv import load_dotenv

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(_repo_root, ".env"))

from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult

from api.agents.retrieval_generation import rag_pipeline
from api.core.embeddings import get_embedding, get_embeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    IDBasedContextPrecision,
    IDBasedContextRecall,
    ResponseRelevancy,
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Pause before each target (RAG) run to reduce rate-limit bursts from HF / Groq / Gemini
RAG_PIPELINE_DELAY_SECONDS = float(os.getenv("RAG_PIPELINE_DELAY_SECONDS", "30"))

ls_client = Client()
qdrant_client = QdrantClient(url=QDRANT_URL)


class AppEmbeddings:
    """LangChain-compatible Embeddings adapter over api.core.embeddings, so RAGAS
    scoring uses whatever EMBEDDING_PROVIDER the rest of the app is configured for
    (openai or huggingface) instead of assuming OpenAI is available."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)


embeddings = AppEmbeddings()

_gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not _gemini_key:
    raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env to use Gemini for RAGAS.")

ragas_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0,
        google_api_key=_gemini_key,
    )
)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)


def _is_rag_output_dict(d: object) -> bool:
    return (
        isinstance(d, dict)
        and "question" in d
        and "answer" in d
        and "retrieved_context" in d
    )


def _target_outputs(run) -> dict:
    """Resolve rag_pipeline dict from Run/RunTree (unwrap LangSmith output nesting + child runs)."""

    def from_outputs_block(out: dict) -> dict:
        if _is_rag_output_dict(out):
            return out
        for k in ("output", "result"):
            inner = out.get(k)
            if _is_rag_output_dict(inner):
                return inner
        return {}

    visited: set[int] = set()

    def walk(node) -> dict:
        if node is None:
            return {}
        nid = id(node)
        if nid in visited:
            return {}
        visited.add(nid)

        out = getattr(node, "outputs", None)
        if isinstance(out, dict):
            got = from_outputs_block(out)
            if got:
                return got
        for child in getattr(node, "child_runs", None) or ():
            got = walk(child)
            if got:
                return got
        return {}

    return walk(run)


def _example_fields(example) -> dict:
    """Merge Example.inputs and Example.outputs so labels are found regardless of storage."""
    if example is None:
        return {}
    merged: dict = {}
    for attr in ("inputs", "outputs"):
        block = getattr(example, attr, None)
        if isinstance(block, dict):
            merged.update(block)
    return merged


def _reference_context_ids(fields: dict) -> list:
    for key in ("reference_context_ids", "chunk_ids", "relevant_context_ids"):
        v = fields.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    return []


def _eval_score(key: str, raw) -> EvaluationResult:
    """Coerce Ragas / NumPy scores to JSON-safe floats; explicit keys help LangSmith experiment columns."""
    if raw is None:
        return EvaluationResult(key=key, score=None, comment="no score")
    try:
        val = raw.item() if hasattr(raw, "item") and callable(raw.item) else raw
        s = float(val)
    except (TypeError, ValueError):
        return EvaluationResult(key=key, score=None, comment=f"non-numeric score: {raw!r}")
    if math.isnan(s):
        return EvaluationResult(key=key, score=None, comment="skipped (NaN or missing reference IDs)")
    return EvaluationResult(key=key, score=s)


def run_rag_with_rate_limit_spacing(inputs: dict):
    """Sleep before each traced RAG call so upstream APIs see lower QPS."""
    if RAG_PIPELINE_DELAY_SECONDS > 0:
        time.sleep(RAG_PIPELINE_DELAY_SECONDS)
    return rag_pipeline(inputs["question"], qdrant_client)


def ragas_faithfulness(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_faithfulness",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        sample = SingleTurnSample(
            user_input=o["question"],
            response=o["answer"],
            retrieved_contexts=o["retrieved_context"],
        )
        scorer = Faithfulness(llm=ragas_llm)
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_faithfulness", asyncio.run(_score()))


def ragas_response_relevancy(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_response_relevancy",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        sample = SingleTurnSample(
            user_input=o["question"],
            response=o["answer"],
            retrieved_contexts=o["retrieved_context"],
        )
        # strictness (default 3) samples multiple candidate generations per
        # call, which Gemini rejects ("Multiple candidates is not enabled for
        # this model") regardless of the LLM wrapper's own n=1.
        scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1)
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_response_relevancy", asyncio.run(_score()))


def ragas_context_precision_id_based(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_context_precision_id_based",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        ref_ids = _reference_context_ids(_example_fields(example))
        if not ref_ids:
            return math.nan
        sample = SingleTurnSample(
            retrieved_context_ids=o["retrieved_context_ids"],
            reference_context_ids=ref_ids,
        )
        scorer = IDBasedContextPrecision()
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_context_precision_id_based", asyncio.run(_score()))


def ragas_context_recall_id_based(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_context_recall_id_based",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )

    async def _score():
        ref_ids = _reference_context_ids(_example_fields(example))
        if not ref_ids:
            return math.nan
        sample = SingleTurnSample(
            retrieved_context_ids=o["retrieved_context_ids"],
            reference_context_ids=ref_ids,
        )
        scorer = IDBasedContextRecall()
        return await scorer.single_turn_ascore(sample)

    return _eval_score("ragas_context_recall_id_based", asyncio.run(_score()))


results = ls_client.evaluate(
    run_rag_with_rate_limit_spacing,
    data="rag-evaluation-dataset",
    evaluators={
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision_id_based,
        ragas_context_recall_id_based,
    },
    experiment_prefix="retriever",
    max_concurrency=1,
)
