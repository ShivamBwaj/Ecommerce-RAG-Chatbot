import asyncio
import math
import os
import sys
import time

# Shift Python path and working directory so that the relative prompt file paths in production code resolve properly.
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, src_path)
os.chdir(src_path)

from langsmith import Client
from langsmith.evaluation.evaluator import EvaluationResult

from api.agents.graph import rag_agent_wrapper

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    IDBasedContextPrecision,
    IDBasedContextRecall,
    ResponseRelevancy,
)

# Environment variables
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Pause before each target (RAG) run to reduce rate-limit bursts from HF / Groq / Gemini
RAG_PIPELINE_DELAY_SECONDS = float(os.getenv("RAG_PIPELINE_DELAY_SECONDS", "30"))

ls_client = Client()
qdrant_client = QdrantClient(url=QDRANT_URL)

model = "sentence-transformers/all-MiniLM-L6-v2"
hf = HuggingFaceEndpointEmbeddings(
    model=model,
    huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
)

groq_api_keys = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY2"),
    os.getenv("GROQ_API_KEY3"),
    os.getenv("GROQ_API_KEY4"),
    os.getenv("GROQ_API_KEY5")
]
valid_groq_keys = [k for k in groq_api_keys if k]

if valid_groq_keys:
    # Groq judge model for RAGAS (no langchain-groq dependency needed).
    from groq import Groq
    from langchain_core.callbacks import Callbacks
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.prompt_values import PromptValue

    class GroqRagasLLM(BaseRagasLLM):
        """Ragas LLM adapter backed by the Groq SDK."""

        def __init__(
            self,
            *,
            api_keys: list[str],
            model: str = "qwen/qwen3-32b",
            run_config=None,
            cache=None,
        ):
            # Ragas expects a real RunConfig; passing None crashes in tenacity retry setup.
            from ragas.run_config import RunConfig

            super().__init__(run_config=run_config or RunConfig(), cache=cache)
            self._api_keys = api_keys
            self._key_index = 0
            self._client = Groq(api_key=self._api_keys[self._key_index])
            self._model = model

        def _rotate_key(self):
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            print(f"Rate limited. Rotating to GROQ_API_KEY index {self._key_index + 1}...")
            self._client = Groq(api_key=self._api_keys[self._key_index])

        def is_finished(self, response: LLMResult) -> bool:
            return True

        def _prompt_to_text(self, prompt: PromptValue) -> str:
            return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

        def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0.01,
            stop=None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            prompt_text = self._prompt_to_text(prompt)
            generations: list[Generation] = []
            for _ in range(n):
                # RAGAS expects strict JSON for many prompts (pydantic parsing).
                # Use a strong system instruction and (when supported) request JSON-only output.
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Return ONLY valid JSON that matches the requested schema. "
                            "Do not add explanations, markdown, or extra text."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ]

                completion = None
                max_attempts = len(self._api_keys) * 3
                for attempt in range(max_attempts):
                    try:
                        completion = self._client.chat.completions.create(
                            model=self._model,
                            messages=messages,
                            temperature=temperature,
                            stop=stop,
                            response_format={"type": "json_object"},
                        )
                        break
                    except Exception as e:
                        err_str = str(e).lower()
                        if "429" in err_str or "rate limit" in err_str:
                            self._rotate_key()
                            if attempt < max_attempts - 1:
                                time.sleep(1)
                                continue
                            raise e
                        
                        # Some Groq models/endpoints may not support response_format; fall back.
                        try:
                            completion = self._client.chat.completions.create(
                                model=self._model,
                                messages=messages,
                                temperature=temperature,
                                stop=stop,
                            )
                            break
                        except Exception as fb_e:
                            fb_err_str = str(fb_e).lower()
                            if "429" in fb_err_str or "rate limit" in fb_err_str:
                                self._rotate_key()
                                if attempt < max_attempts - 1:
                                    time.sleep(1)
                                    continue
                                raise fb_e
                            else:
                                break

                text = completion.choices[0].message.content if completion and completion.choices else ""
                generations.append(Generation(text=text or ""))

            # Ragas pydantic_prompt expects: generations[0][i].text for BaseRagasLLM
            return LLMResult(generations=[generations])

        async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 0.01,
            stop=None,
            callbacks: Callbacks = None,
        ) -> LLMResult:
            return await asyncio.to_thread(
                self.generate_text,
                prompt,
                n=n,
                temperature=temperature,
                stop=stop,
                callbacks=callbacks,
            )

    ragas_llm = GroqRagasLLM(
        api_keys=valid_groq_keys,
        # Default to a generally JSON-compliant chat model; override via env if needed.
        model=os.getenv("GROQ_EVAL_LLM_MODEL2", "llama-3.3-70b-versatile"),
    )
else:
    _gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not _gemini_key:
        raise ValueError(
            "Set GROQ_API_KEYs (preferred) or GOOGLE_API_KEY/GEMINI_API_KEY in your .env to use a judge LLM for RAGAS."
        )

    # Force n=1 to avoid Gemini "Multiple candidates is not enabled for this model".
    ragas_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0,
            google_api_key=_gemini_key,
            n=1,
        )
    )

ragas_embeddings = LangchainEmbeddingsWrapper(hf)


def _is_rag_output_dict(d: object) -> bool:
    return (
        isinstance(d, dict)
        and "answer" in d
        and ("retrieved_context" in d or "used_context" in d)
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
    return rag_agent_wrapper(inputs["question"])


def ragas_faithfulness(run, example):
    o = _target_outputs(run)
    if not _is_rag_output_dict(o):
        return EvaluationResult(
            key="ragas_faithfulness",
            score=None,
            comment="missing RAG outputs on run (check trace / outputs nesting)",
        )
    
    question = o.get("question") or (run.inputs.get("question", "") if hasattr(run, "inputs") and run.inputs else "")
    
    async def _score():
        sample = SingleTurnSample(
            user_input=question,
            response=o["answer"],
            retrieved_contexts=o.get("retrieved_context", o.get("used_context", [])),
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

    question = o.get("question") or (run.inputs.get("question", "") if hasattr(run, "inputs") and run.inputs else "")

    async def _score():
        sample = SingleTurnSample(
            user_input=question,
            response=o["answer"],
            retrieved_contexts=o.get("retrieved_context", o.get("used_context", [])),
        )
        scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
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
    
    question = o.get("question") or (run.inputs.get("question", "") if hasattr(run, "inputs") and run.inputs else "")

    async def _score():
        ref_ids = _reference_context_ids(_example_fields(example))
        if not ref_ids:
            return math.nan
        sample = SingleTurnSample(
            user_input=question,
            response=o["answer"],
            retrieved_context_ids=o.get("retrieved_context_ids", o.get("used_context_ids", [])),
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
    
    question = o.get("question") or (run.inputs.get("question", "") if hasattr(run, "inputs") and run.inputs else "")

    async def _score():
        ref_ids = _reference_context_ids(_example_fields(example))
        if not ref_ids:
            return math.nan
        sample = SingleTurnSample(
            user_input=question,
            response=o["answer"],
            retrieved_context_ids=o.get("retrieved_context_ids", o.get("used_context_ids", [])),
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
