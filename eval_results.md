# Retrieval Evaluation Results

Benchmarked with RAGAS (via `apps/api/evals/eval_retriever2.py`) against a 28-question golden dataset, tracked in LangSmith (`retriever-*` experiments). LLM: Groq (`qwen/qwen3.8-27b`), embeddings: HuggingFace (`all-MiniLM-L6-v2`), judge: Gemini.

## Baseline vs. tuned retrieval

| Metric | Baseline (`top_k=5`) | Tuned (`top_k=3` + `max_tokens` cap) | Change |
|---|---|---|---|
| **Context Precision** (`ragas_context_precision_id_based`) | 0.236 | **0.349** | **+48%** |
| Context Recall (`ragas_context_recall_id_based`) | 1.000 | 0.905 | -9.5% |
| Faithfulness (`ragas_faithfulness`) | 0.688 | 0.727 | +5.7% |
| Response Relevancy (`ragas_response_relevancy`) | 0.876 | 0.738 | -15.8%* |

\* Under active tuning — see note below.

## Root cause and fix

The baseline retrieved `top_k=5` chunks per query, but most questions in the eval dataset have only 1-2 truly relevant items against the reference answer set. With recall already perfect (1.000) at k=5, precision was mathematically capped low (retrieving 5 chunks when only ~1 are relevant caps precision near 0.2) regardless of ranking quality — this wasn't a ranking-quality problem, it was an over-retrieval problem.

**Fix:** reduced `top_k` from 5 to 3 directly in the Qdrant query (`retrieval_generation.py::rag_pipeline`), not by filtering the formatted prompt text after the fact. A first attempt at the latter approach measured *zero* change in precision (0.236 → 0.235) because RAGAS's context-precision metric scores the raw `retrieved_context_ids` returned by the retrieval call, not what ends up in the LLM prompt — an important lesson in verifying which layer a metric actually measures before optimizing it.

## Known tradeoffs

- **Recall dropped** from 1.000 to 0.905 — expected: fewer candidates means occasionally missing the one relevant item that would've been in slots 4-5.
- **Response relevancy dropped** from 0.876 to 0.738. Likely cause: fixing `top_k` surfaced a separate, unrelated issue — Groq's free tier caps output at ~1000 tokens/minute *per key*, and this app's detailed-answer prompt was routinely requesting 1200-1400 output tokens, which fails outright (not just throttles) since a single request already exceeds the per-minute budget. A `max_tokens=800` cap was added to make requests succeed at all, but likely truncates some answers mid-sentence, which RAGAS's relevancy scorer penalizes. Currently retesting with `max_tokens=950` (more headroom, still under the 1000/min ceiling) to see how much of the relevancy score recovers.

## Reproducing

```bash
uv run --package api python apps/api/evals/eval_retriever2.py
```

Requires `GOOGLE_API_KEY`/`GEMINI_API_KEY` (judge, default) or `EVAL_JUDGE_PROVIDER=groq` to force the Groq judge instead (not recommended — measured a near-100% JSON-parse failure rate on the Faithfulness/ResponseRelevancy prompts specifically, not occasional flakiness).
