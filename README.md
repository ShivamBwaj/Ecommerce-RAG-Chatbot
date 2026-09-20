# 🛒 Multi-Agent Ecommerce Shopping Assistant

> A multi-agent system (LangGraph coordinator delegating to specialist agents) with RAG as one of its capabilities — product Q&A and a persistent shopping cart, built with FastAPI, LangGraph, Streamlit, and Qdrant.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-00a393.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1c3c3c.svg)](https://www.langchain.com/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55+-ff4b4b.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-4f8bc8.svg)](https://qdrant.tech/)
[![Render](https://img.shields.io/badge/Deployed-Render-46e3b7.svg)](https://render.com/)

**Live demo:** [Streamlit UI](https://handson-streamlit.onrender.com) · [API](https://handson-api.onrender.com) — free tier, spins down on idle so the first request may take 30-50s to wake up.

## 📋 Overview

A shopping assistant for a music/CDs/vinyl catalog. It answers product questions (specs, reviews, recommendations) and manages a real, persisted shopping cart (add/remove/view items) — routed through a coordinator agent that delegates to whichever specialist the query needs, and can chain both in a single turn ("find me a Queen vinyl and add it to my cart").

### ✨ Key Features

- **🧑‍🤝‍🧑 Multi-agent coordinator**: a `coordinator_agent` plans and delegates to a `product_qa_agent` (catalog search, reviews) and a `shopping_cart_agent` (add/remove/view cart), each with their own tools and iteration budget
- **🛒 Persistent cart**: cart state lives in Postgres (`shopping_carts.shopping_cart_items`), survives restarts, scoped per `user_id`/`cart_id`
- **💬 Multi-turn memory**: conversation + agent state checkpointed per `thread_id` via LangGraph's `PostgresSaver`
- **🔍 Hybrid Search**: dense (vector) + BM25 sparse search fused with Reciprocal Rank Fusion (RRF) in Qdrant
- **📡 Streaming UX**: SSE stream shows live progress ("Planning...", "Looking for items: ...") before the final answer
- **📊 RAG Evaluation**: RAGAS-based benchmark (Faithfulness, Response Relevancy, Context Precision/Recall) tracked in LangSmith
- **🐳 Docker Support**: full local stack via `docker-compose` (api, streamlit, Qdrant, Postgres, MCP servers)
- **☁️ Cloud-deployed**: API + Streamlit on Render, vectors on Qdrant Cloud, LLM on Groq — entirely on free tiers
- **🔧 Provider-agnostic**: `EMBEDDING_PROVIDER` and LLM provider both switch between OpenAI and a fully free Groq + HuggingFace stack via config, no code changes

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────┐
│   Streamlit     │────▶│   FastAPI (LangGraph multi-agent)         │
│   Frontend      │     │                                            │
│  (Render)       │◀────│   coordinator_agent                       │
└─────────────────┘     │      ├─▶ product_qa_agent ─▶ Qdrant tools  │
                         │      └─▶ shopping_cart_agent ─▶ Postgres  │
                         │   (Render)                                │
                         └───────────────┬────────────────────────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        ▼                 ▼                 ▼
                 ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
                 │ Qdrant Cloud│  │  Postgres     │  │ Groq / HF     │
                 │ (vectors)   │  │ (Render, cart │  │ (LLM + embed) │
                 │             │  │ + checkpoints)│  │               │
                 └─────────────┘  └──────────────┘  └───────────────┘
```

### Request flow

1. User sends a message from Streamlit (with `thread_id`, `user_id`, `cart_id`)
2. `coordinator_agent` reads the conversation, decides whether this needs `product_qa_agent`, `shopping_cart_agent`, both, or neither
3. Each worker agent calls its own tools (`get_formatted_items_context` / `get_formatted_reviews_context` for product QA; `add_to_shopping_cart` / `get_shopping_cart` / `remove_from_cart` for the cart) and loops back to the coordinator with results
4. Coordinator repeats until it has enough information, then sets `final_answer=True`
5. Response streams back over SSE; the API resolves product IDs to images/prices from Qdrant for the UI's product cards

## 🚀 Quick Start (local)

### Prerequisites

- **Python 3.12+**, [uv](https://docs.astral.sh/uv/) for dependency management
- **Docker Desktop** (for the full local stack)
- API keys: `GROQ_API_KEY` (LLM), `HF_API_TOKEN` (embeddings) — both free tier. `OPENAI_API_KEY` optional if you prefer OpenAI for both.

### 1. Configure environment

Copy `.env.example` to `.env` and fill in your keys. The free stack (default) needs:

```env
GROQ_API_KEY=your_groq_key
EMBEDDING_PROVIDER=huggingface
HF_API_TOKEN=your_hf_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=amazon-items-collection-01-hybrid-search
```

Add more `GROQ_API_KEY2`..`GROQ_API_KEY7` for round-robin rotation if you're hitting rate limits (`api/core/llm.py` picks a different key per LLM call).

### 2. Run everything

```bash
docker compose up --build
```

This starts `qdrant`, `postgres` (auto-runs `scripts/sql/shopping_cart_table.sql` on first init), `api`, `streamlit-app`, and the two MCP servers.

- 🖥️ UI: http://localhost:8501
- 🔌 API: http://localhost:8000
- 💾 Qdrant: http://localhost:6333

The API self-provisions its LangGraph checkpoint tables on startup — no manual migration needed for a fresh Postgres.

### 3. Load product data

If starting from scratch, `apps/api/scripts/reindex_openai_embeddings.py` re-embeds and indexes the catalog. To copy an existing Qdrant instance to a fresh one (e.g., migrating to Qdrant Cloud), use `apps/api/scripts/migrate_to_qdrant_cloud.py` instead — it copies vectors directly rather than re-embedding.

## ☁️ Deployment (Render + Qdrant Cloud)

The live deployment uses:
- **Render** — `api` and `streamlit-app` as native Python web services (not Docker: Render's CLI has no way to point at a Dockerfile in a subdirectory with a repo-root build context, so these run via `uv sync` + `uv run uvicorn`/`streamlit run` directly), plus a managed Postgres
- **Qdrant Cloud** — free 1GB cluster; `apps/api/scripts/migrate_to_qdrant_cloud.py` handles the one-time migration from local
- `render.yaml` documents the equivalent Blueprint shape, though the live services were provisioned via `render services create` (see file header for why)

**Known free-tier constraints:**
- Render web services sleep after inactivity (cold start ~30-50s)
- Render free Postgres **expires 30 days after creation** and needs manual renewal in the dashboard
- Groq free tier caps output at ~1000 tokens/minute *per key* — a single detailed response can exceed that on one key alone, which is why the app rotates across multiple keys and caps `max_tokens`

## 📊 Evaluation Results

Benchmarked with **RAGAS** against a 28-question golden dataset, tracked in LangSmith. Two rounds, since the first attempt at improving retrieval precision targeted the wrong layer (see below):

| Metric | Baseline (`top_k=5`) | Tuned (`top_k=3` + token cap) |
|---|---|---|
| **Context Precision** | 0.236 | **0.349 (+48%)** |
| Context Recall | 1.000 | 0.905 |
| Faithfulness | 0.688 | 0.727 |
| Response Relevancy | 0.876 | 0.738* |

\* Recovering this is an active area — see `apps/api/evals/eval_retriever2.py` and `retrieval_generation.py` for the current `max_tokens` tuning.

**Root cause diagnosis:** the baseline retrieved `top_k=5` chunks per query, but most questions in the eval set have only 1-2 truly relevant items — precision was mathematically capped low regardless of ranking quality, while recall was already perfect. The fix was reducing `top_k` at the actual Qdrant query (`retrieval_generation.py::rag_pipeline`), not just trimming what got formatted into the prompt — an important first attempt that measured no change, because it filtered *after* retrieval instead of retrieving less.

### Running the benchmark

```bash
uv run --package api python apps/api/evals/eval_retriever2.py
```

Uses a Gemini judge by default for RAGAS scoring (`EVAL_JUDGE_PROVIDER=groq` to force Groq instead — found to fail ~100% of the time on the more complex Faithfulness/ResponseRelevancy prompts, not just occasionally). Results post to LangSmith under the `retriever-*` experiment prefix.

## 🗂️ Project Structure

```
.
├── apps/
│   ├── api/
│   │   ├── src/api/
│   │   │   ├── app.py                     # FastAPI app; self-provisions checkpoint tables on startup
│   │   │   ├── api/                       # endpoints, request/response models, middleware
│   │   │   ├── agents/
│   │   │   │   ├── graph.py                # coordinator/product_qa/shopping_cart LangGraph workflow
│   │   │   │   ├── agents.py               # agent node functions + response models
│   │   │   │   ├── tools.py                # Qdrant retrieval + shopping cart tools
│   │   │   │   ├── retrieval_generation.py # single-shot RAG pipeline (used by evals)
│   │   │   │   └── prompts/                # per-agent YAML prompt templates
│   │   │   └── core/                       # config, LLM client (with key rotation), embeddings
│   │   ├── evals/                          # RAGAS benchmark scripts
│   │   ├── scripts/                        # reindexing + Qdrant Cloud migration
│   │   └── Dockerfile
│   ├── chatbot-ui/                         # Streamlit frontend
│   ├── items_mcp_server/                   # standalone MCP server exposing item retrieval
│   └── reviews_mcp_server/                 # standalone MCP server exposing review retrieval
├── notebooks/                              # week-by-week bootcamp notebooks (source of truth
│                                            # for what's since been ported into apps/api)
├── scripts/sql/shopping_cart_table.sql     # cart schema (auto-applied via docker-entrypoint-initdb.d)
├── .github/workflows/ci.yml                # import smoke-checks + docker builds per app
├── render.yaml                             # documents the Render deployment shape
├── docker-compose.yaml
└── .env.example
```

## 🔧 API Reference

### `POST /rag/`

```json
{
  "query": "search your catalog for Queen albums on vinyl",
  "thread_id": "conversation-id",
  "user_id": "user-id",
  "cart_id": "cart-id"
}
```

Returns a `text/event-stream` of progress messages, ending with:

```json
{
  "type": "final_result",
  "data": {
    "answer": "...",
    "used_context": [{"image_url": "...", "price": 29.33, "description": "..."}],
    "trace_id": "..."
  }
}
```

### `POST /submit_feedback/`

Submits thumbs up/down + optional text feedback for a trace, forwarded to LangSmith.

## 🐛 Troubleshooting

- **`No user query found in messages`** (Groq only): every LLM call needs at least one `user`-role message — a system-only prompt that OpenAI tolerates gets rejected outright by Groq's chat template.
- **Qdrant `400 Bad Request: Index required but not found`**: Qdrant Cloud enforces payload indexes for filtered fields that local Qdrant doesn't require. Create one with `PUT /collections/{name}/index` (`field_schema: "keyword"`) for any field you filter on (e.g. `parent_asin`).
- **Groq `organization_restricted`**: account-level, not fixable in code — remove that key from rotation and check the Groq console.
- **`psql`/direct Postgres connection resets over a VPN**: if you're behind something like Cloudflare WARP, raw TCP+TLS to a managed Postgres can fail unpredictably even though the port is reachable. Prefer running one-off admin scripts *from* the deployed service (`render ssh`) over a flaky local network path.

## 📚 Data

Amazon CDs & Vinyl catalog — product metadata, reviews, ratings, images. Catalog is music-only by design (the agent prompt explicitly declines out-of-catalog requests rather than hallucinating availability).

## 🤝 Contributing

Bootcamp project. Notebooks under `notebooks/weekN/` are the working/exploratory versions of what eventually gets ported into `apps/api` — check there first if you're extending agent behavior, since that's usually where a feature gets prototyped before it's productionized.

## 📄 License

Educational project — no license specified.

---

**Built as part of the AI Engineering Bootcamp**
