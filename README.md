# 🛒 Ecommerce RAG Chatbot

> A production-ready Retrieval-Augmented Generation (RAG) chatbot for ecommerce product recommendations, built with FastAPI, Streamlit, and Qdrant vector database.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55+-ff4b4b.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-4f8bc8.svg)](https://qdrant.tech/)

## 📋 Overview

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** chatbot specifically designed for ecommerce product discovery and recommendations. The chatbot can answer natural language questions about products in the catalog, provide detailed product information, and suggest relevant items with images and pricing.

### ✨ Key Features

- **🔍 Hybrid Search**: Combines semantic (vector) search with BM25 keyword search using Reciprocal Rank Fusion (RRF)
- **🧠 LLM-Powered Responses**: Uses OpenAI by default when `OPENAI_API_KEY` is set, with Groq kept as a fallback
- **📊 Structured Output**: Returns responses with explicit product references, descriptions, and metadata
- **🎯 RAG Evaluation**: Comprehensive evaluation framework using RAGAS metrics (Faithfulness, Relevance, Context Precision/Recall)
- **🐳 Docker Support**: Complete containerized deployment with docker-compose
- **🌐 Modern UI**: Streamlit-based chat interface with product suggestion sidebar
- **🔧 Multi-LLM Support**: Backend supports both Groq and Gemini providers (extensible)

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   FastAPI        │────▶│   Qdrant        │
│   Frontend      │     │   Backend        │     │   Vector DB    │
│   (Port 8501)   │◀────│   (Port 8000)    │◀────│   (Port 6333)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ OpenAI / Groq /  │
                       │   Gemini LLM API │
                       │   LLM API        │
                       └──────────────────┘
```

### Data Flow

1. User submits query via Streamlit UI
2. FastAPI endpoint receives request
3. **Retrieval Phase**:
   - Query embedding via OpenAI `text-embedding-3-small`
   - Hybrid search: Semantic (vector) + BM25 keyword search with RRF fusion
   - Top-k most relevant product chunks retrieved from Qdrant
4. **Generation Phase**:
   - Context formatted with product IDs, ratings, and descriptions
   - Prompt engineered to produce detailed, structured answers
   - LLM generates response with referenced product IDs
5. **Response**:
   - Answer text with detailed product specifications
   - List of used products with images and prices (extracted from Qdrant)
   - Streamlit displays answer and product cards in sidebar

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (recommended: use [uv](https://docs.astral.sh/uv/) for dependency management)
- **Docker Desktop** (optional, for containerized deployment)
- **API Keys**:
    - `OPENAI_API_KEY` (required for the default LLM path) - from [OpenAI](https://platform.openai.com/)
    - `GROQ_API_KEY` (optional fallback)
    - `GEMINI_API_KEY` or `GOOGLE_API_KEY` (optional, for evaluation)

### 1. Clone and Setup

```bash
# Clone the repository
cd "C:\Users\Loq\Documents\CRAP\end to end aibootcamp\code\handsON"

# Install dependencies (using uv)
uv sync
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Required: OpenAI API key for production chatbot
OPENAI_API_KEY=your_openai_api_key_here

# Optional fallback
GROQ_API_KEY=your_groq_api_key_here

# Optional: For evaluation (RAGAS uses Gemini)
GEMINI_API_KEY=your_gemini_api_key_here
# OR
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Hugging Face token if you switch EMBEDDING_PROVIDER back to huggingface
HF_API_TOKEN=your_hf_token_here

# Embeddings
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_DENSE_VECTOR_NAME=all-MiniLM-L6-v2

# Qdrant connection (use localhost for local, 'qdrant' for docker)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=amazon-items-collection-02-openai-small
QDRANT_SPARSE_VECTOR_NAME=bm25

# API endpoint for Streamlit UI
API_URL=http://localhost:8000
```

### 3. Start Qdrant Vector Database

**Option A: Docker (Recommended)**

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Option B: Local binary** - Download from [qdrant.tech](https://qdrant.tech/documentation/quick-start/)

### 4. Load Product Data into Qdrant

```bash
# Activate virtual environment if using uv
uv sync

PYTHONPATH=apps/api/src QDRANT_URL=http://localhost:6333 \
  python apps/api/scripts/reindex_openai_embeddings.py --recreate
```

### 5. Run the Application

**Option A: Docker Compose (All-in-One)**

```bash
docker compose up --build
```

Services:
- 🖥️ **UI**: http://localhost:8501
- 🔌 **API**: http://localhost:8000
- 💾 **Qdrant**: http://localhost:6333

**Option B: Local Development**

Terminal 1 - Start API:
```bash
uv run --package api uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 - Start UI:
```bash
uv run --package chatbot-ui streamlit run apps/chatbot-ui/src/chatbot_ui/app.py
```

Open http://localhost:8501 in your browser.

## 📊 Evaluation Results

The system has been evaluated on **28 test samples** using the **RAGAS** framework (Retrieval-Augmented Generation Assessment). Here are the metrics:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Faithfulness** (ragas_faithfulness) | **0.881** | High - answers are grounded in retrieved context |
| **Response Relevancy** (ragas_response_relevancy) | **0.841** | High - answers are relevant to user queries |
| **Context Recall** (ragas_context_recall_id_based) | **0.957** | Very High - system retrieves most relevant items |
| **Context Precision** (ragas_context_precision_id_based) | **0.217** | Moderate - precision could be improved |
| **Median Latency** | **31.52s** | Includes 45s rate-limit delay for Groq API |
| **Total Cost** | **$0.00** | Evaluations run on Groq (free tier) + Gemini for scoring |

### 📈 Understanding the Metrics

- **Faithfulness (0.881)**: The chatbot rarely hallucinates; it sticks to retrieved product data
- **Response Relevancy (0.841)**: User questions are answered appropriately
- **Context Recall (0.957)**: Excellent - finds most of the truly relevant products
- **Context Precision (0.217)**: Lower precision indicates the retriever brings in some irrelevant items alongside relevant ones (trade-off for high recall)
- **Latency Note**: The 31.52s median latency **includes a 45s artificial delay** (`RAG_PIPELINE_DELAY_SECONDS=45`) to avoid hitting Groq's rate limits during evaluation. In production without this delay, latency would be ~2-5s.

### 🔬 Running Evaluations

```bash
# Set rate-limit delay (adjust as needed)
export RAG_PIPELINE_DELAY_SECONDS=45  # or 0 for no delay

# Run evaluation
make run-evals-retriever
# or manually:
uv run --env-file .env python apps/api/evals/eval_retriever.py
```

Results are uploaded to **LangSmith** for tracking and analysis. View experiments at: https://smith.langchain.com/

## 🔧 API Reference

### `POST /rag`

Generate a chatbot response for a user query.

**Request:**
```json
{
  "query": "I'm looking for a jazz album with piano improvisations"
}
```

**Response:**
```json
{
  "request_id": "abc-123-def",
  "answer": "Based on your query, I found a jazz album that matches...",
  "used_context": [
    {
      "image_url": "https://m.media-amazon.com/images/I/517h9OROQAL.jpg",
      "price": 14.98,
      "description": "Solo acoustic fingerstyle guitar."
    }
  ]
}
```

**Headers:**
- `X-Request-ID`: Unique request identifier for tracing

**Status:** 200 OK

## 🗂️ Project Structure

```
.
├── apps/
│   ├── api/                          # FastAPI backend
│   │   ├── src/api/
│   │   │   ├── app.py               # FastAPI app + middleware
│   │   │   ├── api/
│   │   │   │   ├── endpoints.py     # /rag endpoint
│   │   │   │   ├── models.py        # Pydantic request/response models
│   │   │   │   └── middleware.py    # Request ID middleware
│   │   │   ├── agents/
│   │   │   │   ├── retrieval_generation.py  # Core RAG pipeline
│   │   │   │   └── prompts/
│   │   │   │       └── retrieval_generation.yaml  # LLM prompt template
│   │   │   └── core/
│   │   │       └── config.py        # Configuration from .env
│   │   ├── Dockerfile
│   │   └── pyproject.toml
│   │
│   └── chatbot-ui/                   # Streamlit frontend
│       ├── src/chatbot_ui/
│       │   └── app.py               # Main Streamlit app
│       │   └── core/
│       │       └── config.py        # UI configuration
│       ├── Dockerfile
│       └── pyproject.toml
│
├── data/                            # Product catalog (Amazon CDs & Vinyl)
│   ├── CDs_and_Vinyl.jsonl          # Product metadata and reviews
│   └── meta_CDs_and_Vinyl.jsonl     # Structured product metadata
│
├── evals/                           # Evaluation framework
│   ├── eval_retriever.py           # RAGAS evaluation script
│   └── eval_results.md             # Evaluation metrics and results
│
├── qdrant_storage/                  # Qdrant persistence (gitignored)
├── docker-compose.yaml              # Orchestrates all services
├── Makefile                         # Convenience commands
├── pyproject.toml                   # Root workspace config (uv)
├── .env.example                     # Environment template
└── README.md                        # This file
```

## 🧠 How It Works: The RAG Pipeline

### Step-by-Step

```python
# Simplified pipeline flow
query = "Recommend a classical piano album"

# 1. RETRIEVE
embedding = hugging_face.embed(query)  # → [384-dim vector]
results = qdrant.hybrid_search(
    vector=embedding,
    keyword=query,  # BM25
    fusion="rrf"
)
# Returns: top 5 product chunks with IDs, descriptions, ratings

# 2. FORMAT CONTEXT
context = "\n".join([
    f"- ID: {item.id}, rating: {item.rating}, description: {item.desc}"
    for item in results
])

# 3. GENERATE PROMPT
prompt = prompt_template.render(
    preprocessed_context=context,
    question=query
)

# 4. CALL LLM
response = groq.chat.completions.create(
    model="qwen/qwen3-32b",
    response_model=RAGGenerationResponse,  # Structured output via instructor
    messages=[{"role": "system", "content": prompt}]
)

# 5. EXTRACT PRODUCTS & RENDER
answer = response.answer
references = response.references  # List of {id, description}

# 6. FETCH PRODUCT DETAILS (images, prices)
for ref in references:
    product = qdrant.get_by_id(ref.id)
    used_context.append({
        "image_url": product.image,
        "price": product.price,
        "description": ref.description
    })
```

### Core Components

#### 🔍 Hybrid Retrieval (`retrieval_generation.py:90-131`)

The system uses **Qdrant's hybrid search** with:

- **Dense vectors** (`text-embedding-3-small` via OpenAI): Semantic similarity
- **Sparse vectors** (BM25): Keyword matching
- **Reciprocal Rank Fusion (RRF)**: Combines both result sets for optimal recall

**Configuration:**
```python
prefetch=[
    Prefetch(
        query=query_embedding,
        using="text-embedding-3-small",
        limit=20  # Recall candidates from vector search
    ),
    Prefetch(
        query=Document(text=query, model="qdrant/bm25"),
        using="bm25",
        limit=20  # Recall candidates from keyword search
    )
],
query=FusionQuery(fusion="rrf"),
limit=5  # Final top-k
```

#### 🎯 Structured LLM Output

Uses **Instructor** to enforce a Pydantic schema:

```python
class RAGUsedContext(BaseModel):
    id: str
    description: str

class RAGGenerationResponse(BaseModel):
    answer: str
    references: list[RAGUsedContext]
```

This guarantees the LLM returns **exact product IDs** that can be looked up for images/prices.

#### 💬 Prompt Engineering

The prompt template (`retrieval_generation.yaml`) instructs the LLM to:

1. Answer based **only** on provided context (no hallucinations)
2. Return **detailed specifications** in bullet points
3. Provide **short descriptions** of referenced products
4. **Never expose** product IDs in the final answer (only to the system)

### Model Choice: Why `qwen/qwen3-32b`?

- ✅ **Speed**: Groq's LPU inference provides ~200 tokens/sec
- ✅ **Quality**: 32B parameter model with reasoning capabilities
- ✅ **Cost**: Currently free tier on Groq
- ✅ **Structured Output**: Instructor integration works reliably

## ⚙️ Configuration & Environment

### All Environment Variables

OpenAI embeddings are the default retrieval path:

```env
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
QDRANT_COLLECTION=amazon-items-collection-02-openai-small
QDRANT_SPARSE_VECTOR_NAME=bm25
```

The old Hugging Face MiniLM settings are still available only if you explicitly set `EMBEDDING_PROVIDER=huggingface`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Groq Cloud API key |
| `GEMINI_API_KEY` | ❌ No | — | Gemini API key (or use `GOOGLE_API_KEY`) |
| `GOOGLE_API_KEY` | ❌ No | — | Alternative to `GEMINI_API_KEY` |
| `HF_API_TOKEN` | ⚠️ Recommended | — | Hugging Face token for higher rate limits |
| `HF_EMBEDDING_MODEL` | ❌ No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `QDRANT_URL` | ❌ No | `http://qdrant:6333` | Qdrant connection URL |
| `API_URL` | ❌ No | `http://api:8000` | Backend URL for frontend |
| `RAG_PIPELINE_DELAY_SECONDS` | ❌ No | `30` | Rate-limit delay before RAG calls |

### Docker vs Local

- **Docker Compose**: `QDRANT_URL=http://qdrant:6333`, `API_URL=http://api:8000`
- **Local Development**: `QDRANT_URL=http://localhost:6333`, `API_URL=http://localhost:8000`

## 🧪 Testing & Evaluation

### Unit/Integration Testing

The project uses **RAGAS** for end-to-end evaluation:

```bash
# Run with 45s delay to respect Groq rate limits
RAG_PIPELINE_DELAY_SECONDS=45 make run-evals-retriever
```

**Test Dataset:** 28 curated queries with expected product references
**Framework:** RAGAS 0.4.3
**Evaluation LLM:** Gemini 1.5 Flash Lite (for scoring)
**Scoring Metrics:**

1. **Faithfulness** - Does the answer stick to retrieved context?
2. **Response Relevancy** - Is the answer relevant to the query?
3. **Context Recall** - Are all relevant products retrieved?
4. **Context Precision** - Are retrieved products actually relevant?

### Manual Testing

```bash
# Start the UI and try these queries:
- "Recommend a jazz piano album"
- "Find classical music under $20"
- "I want a blues CD with high ratings"
- "Show me folk music suitable for relaxation"

# Check logs for performance:
tail -f logs/api.log
```

## 📈 Performance Considerations

### Latency Breakdown

| Component | Approx. Time |
|-----------|--------------|
| OpenAI embedding | 300ms - 1s |
| Qdrant hybrid search | 100-300ms |
| Groq LLM inference | 1-2s |
| Structured output parsing | 100ms |
| **Total (without delays)** | **~2-4s** |
| **Total (with rate-limit delay)** | **~45-50s** |

### Rate Limit Handling

During evaluation, we deliberately add a delay to avoid hitting Groq's rate limits:

```python
RAG_PIPELINE_DELAY_SECONDS = float(os.getenv("RAG_PIPELINE_DELAY_SECONDS", "30"))

def run_rag_with_rate_limit_spacing(inputs: dict):
    if RAG_PIPELINE_DELAY_SECONDS > 0:
        time.sleep(RAG_PIPELINE_DELAY_SECONDS)
    return rag_pipeline(inputs["question"], qdrant_client)
```

Set `RAG_PIPELINE_DELAY_SECONDS=0` for production use with proper rate-limit handling (retries with exponential backoff recommended).

## 🔒 Security & Best Practices

- ✅ **Secrets management**: `.env` file gitignored
- ✅ **Pydantic validation**: Request/response models validated
- ✅ **CORS configured**: Adjust `allow_origins` for production
- ✅ **Request ID tracing**: `X-Request-ID` passed through all logs
- ✅ **Structured logging**: JSON-formatted logs with request context
- ✅ **LangSmith tracing**: All LLM calls traced for monitoring

## 🐛 Troubleshooting

### Qdrant Connection Error
```bash
# Check if Qdrant is running
curl http://localhost:6333/healthz

# If using Docker, verify container:
docker ps | grep qdrant
```

### Groq Rate Limits
- Reduce batch size or add delays: `RAG_PIPELINE_DELAY_SECONDS=60`
- Check quota: https://console.groq.com/settings/limits

### Hugging Face Timeout
```python
# Increase timeout in retrieval_generation.py:
with urlopen(request, timeout=120) as response:  # Currently 120s
```

### Docker Compose Services Not Starting
```bash
# Rebuild with no cache
docker compose up --build --force-recreate
```

## 📚 Data: Amazon CDs & Vinyl Dataset

The chatbot is trained on the **Amazon CDs & Vinyl** dataset (~1.5M products). The data includes:

- Product metadata (title, artist, price, category, description)
- Customer reviews and ratings
- Product images (various resolutions)
- ASIN identifiers (Amazon Standard Identification Numbers)

**Sample Products:**
- Classical piano recordings
- Jazz improvisation albums
- Soundtrack collections
- International music
- Genre: Rock, Pop, Hip-Hop, Electronic, Folk, etc.

This diverse catalog makes for interesting conversational queries across multiple music genres.

## 🚧 Future Improvements

1. **🔄 Query Rewriting**: Improve retrieval for complex multi-intent queries
2. **🎵 Audio Previews**: Integrate 30-second audio samples in UI
3. **📱 Mobile Responsive**: Better UI for smaller screens
4. **🔍 Faceted Search**: Filter by genre, price range, rating, release date
5. **👤 User Personalization**: Track user preferences and browsing history
6. **💾 Caching Layer**: Redis caching for frequent queries
7. **🧠 Better Embeddings**: Fine-tuned embedding model on music domain
8. **📊 Analytics Dashboard**: Monitoring usage patterns, popular queries
9. **🎯 A/B Testing**: Compare different LLM providers/models
10. **⚡ Async API**: Async endpoints for higher throughput

## 🤝 Contributing

This is a bootcamp project. To extend:

1. Add new LLM providers (OpenAI, Anthropic, etc.)
2. Experiment with different embedding models
3. Improve the prompt template with few-shot examples
4. Add unit tests for the retrieval pipeline
5. Implement rate-limit handling (retries, circuit breaker)

## 📄 License

Educational project - no license specified.

## 🙏 Acknowledgments

- **Dataset**: Amazon product data (publicly available)
- **LLM**: Groq for fast, free inference
- **Vector DB**: Qdrant for hybrid search
- **UI**: Streamlit for rapid prototyping
- **Framework**: FastAPI for robust API

---

**Built with ❤️ as part of the AI Engineering Bootcamp**

*Questions or feedback? Open an issue or reach out!*
