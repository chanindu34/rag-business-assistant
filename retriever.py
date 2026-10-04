"""
Production RAG retriever.

Per query:
1. Semantic router: decide whether HyDE is worth an LLM call.
2. HyDE (optional): draft a hypothetical answer and embed it for dense search.
3. Hybrid search: BM25 (keywords) + dense (cosine), fused with
   Reciprocal Rank Fusion (RRF). RRF uses ranks, not raw scores, so the two
   retrievers' very different score scales never need normalising.
4. Cross-encoder reranking of the fused candidates.
5. Confidence tiers on the top rerank score: high / ambiguous / low.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from google.genai import types
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from confidence_tiers import ConfidenceTier, ConfidenceTierRouter
from semantic_router import SemanticRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer shared by BM25 indexing and querying
# ---------------------------------------------------------------------------
# Numbers keep their decimals and percent sign ("80.01", "75%"); words are
# lowercased and split from punctuation, so "EBITDA?" matches "EBITDA".
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?|[a-z][a-z0-9]*")
STOPWORDS = frozenset(
    "a an and are as at be been by can did do does for from had has have how i if in "
    "into is it its me my of on or our over so than that the their them there these "
    "they this those to under up was we were what when where which while who whom "
    "why will with would you your".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------
class HyDETransformer:
    """Hypothetical Document Embeddings: embed a drafted answer, not the bare query."""

    PROMPT = """Generate a plausible, detailed answer to this question.
Be specific, as if answering based on a document.
Only generate the answer, no preamble.

Question: {query}

Answer:"""

    def __init__(self, llm_client, generation_model: str, generate_fn=None, enabled: bool = True):
        self.llm_client = llm_client
        self.generation_model = generation_model
        # generate_fn(prompt) -> str routes HyDE through the quota-aware gateway.
        self.generate_fn = generate_fn
        self.enabled = enabled

    def generate_hypothetical_answer(self, query: str) -> str:
        if not self.enabled:
            return query
        prompt = self.PROMPT.format(query=query)
        try:
            if self.generate_fn is not None:
                text = self.generate_fn(prompt)
            else:
                response = self.llm_client.models.generate_content(
                    model=self.generation_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.7),
                )
                text = response.text
            text = (text or "").strip()
            return text or query
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using original query.")
            return query


# ---------------------------------------------------------------------------
# Hybrid retrieval with RRF
# ---------------------------------------------------------------------------
def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each item, best score = rank 1."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


class HybridRetriever:
    """BM25 + dense retrieval fused with weighted Reciprocal Rank Fusion.

    fused(d) = w_bm25 / (k + rank_bm25(d)) + w_dense / (k + rank_dense(d))
    k = 60 is the constant from the original RRF paper (Cormack et al., 2009).
    """

    def __init__(self, chunks: List[str], embeddings, rrf_k: int = 60):
        self.chunks = list(chunks)
        self.rrf_k = rrf_k

        emb = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        zero_rows = int((norms[:, 0] == 0).sum())
        if zero_rows:
            logger.warning(f"{zero_rows} chunk embeddings are zero vectors; they will never match densely.")
        norms[norms == 0] = 1.0
        self.embeddings = emb / norms  # unit vectors, so dot product = cosine

        tokenized = [tokenize(c) or ["<empty>"] for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"Initialized HybridRetriever with {len(self.chunks)} chunks")

    def retrieve(
        self,
        query_text: str,
        query_embedding,
        k: int = 20,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ) -> Tuple[List[str], np.ndarray]:
        n = len(self.chunks)
        fused = np.zeros(n, dtype=np.float64)

        # Sparse: only documents sharing at least one keyword get BM25 credit.
        q_tokens = tokenize(query_text)
        if q_tokens:
            bm25_scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float64)
            if bm25_scores.max() > 0:
                contrib = bm25_weight / (self.rrf_k + _ranks(bm25_scores))
                contrib[bm25_scores <= 0] = 0.0
                fused += contrib

        # Dense: skipped (not crashed) on a zero or wrong-sized query vector.
        q = np.asarray(query_embedding, dtype=np.float32) if query_embedding is not None else np.zeros(0)
        q_norm = float(np.linalg.norm(q)) if q.size else 0.0
        if q_norm > 0 and q.shape[0] == self.embeddings.shape[1]:
            sims = self.embeddings @ (q / q_norm)
            fused += vector_weight / (self.rrf_k + _ranks(sims))
        else:
            logger.warning("Query embedding is zero or the wrong size; dense retrieval skipped.")

        top = np.argsort(-fused, kind="stable")[:k]
        return [self.chunks[i] for i in top], fused[top]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------
class RerankerFilter:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        try:
            self.reranker = CrossEncoder(model_name, max_length=512)
            logger.info(f"Loaded reranker: {model_name}")
        except Exception as e:
            # Fail loudly. A missing reranker used to fall back to score=1.0,
            # which silently marked every answer as HIGH confidence.
            raise RuntimeError(f"Failed to load reranker {model_name}: {e}") from e

    def rerank(self, query: str, candidates: List[str], top_k: int = 3) -> Tuple[List[Dict], float]:
        if not candidates:
            return [], 0.0
        start = time.time()
        try:
            scores = self.reranker.predict([[query, c] for c in candidates])
            latency = (time.time() - start) * 1000
            order = np.argsort(scores)[::-1][:top_k]
            return [
                {"chunk": candidates[i], "rerank_score": float(scores[i]), "rank": j}
                for j, i in enumerate(order)
            ], latency
        except Exception as e:
            # Score 0.0, not 1.0: an unscored answer must land in the LOW tier.
            logger.error(f"Reranking failed: {e}. Marking results as unscored.")
            return [{"chunk": c, "rerank_score": 0.0, "rank": i}
                    for i, c in enumerate(candidates[:top_k])], 0.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
class ProductionRAG:
    def __init__(
        self,
        chunks: List[str],
        embeddings,
        embedding_model: str,
        generation_model: str,
        llm_client,
        reranker_model: str = "BAAI/bge-reranker-base",
        high_threshold: float = 0.6,
        low_threshold: float = 0.35,
        hyde_generate_fn=None,
        use_hyde: bool = True,
        router: Optional[SemanticRouter] = None,
        num_candidates: int = 20,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
    ):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.num_candidates = num_candidates
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self.router = router or SemanticRouter()
        self.hyde = HyDETransformer(llm_client, generation_model, generate_fn=hyde_generate_fn, enabled=use_hyde)
        self.retriever = HybridRetriever(chunks, embeddings, rrf_k=rrf_k)
        self.reranker = RerankerFilter(reranker_model)
        self.tiers = ConfidenceTierRouter(high_threshold=high_threshold, low_threshold=low_threshold)

    def _embed(self, text: str):
        return self.llm_client.models.embed_content(
            model=self.embedding_model, contents=text
        ).embeddings[0].values

    def retrieve(self, query: str, top_k: int = 3, verbose: bool = False) -> Dict:
        # 1. Route
        route = self.router.classify(query) if self.hyde.enabled else "hyde_disabled"

        # 2. HyDE (only when routed to it)
        start = time.time()
        dense_text = self.hyde.generate_hypothetical_answer(query) if route == "run_hyde" else query
        hyde_used = dense_text != query
        hyde_latency = (time.time() - start) * 1000

        # 3. Hybrid: BM25 always sees the real query, dense sees the HyDE draft
        start = time.time()
        candidates, _ = self.retriever.retrieve(
            query, self._embed(dense_text), k=self.num_candidates,
            bm25_weight=self.bm25_weight, vector_weight=self.vector_weight,
        )
        hybrid_latency = (time.time() - start) * 1000

        # 4. Rerank against the real query
        reranked, rerank_latency = self.reranker.rerank(query, candidates, top_k=top_k)

        # 5. Confidence tier
        top_score = reranked[0]["rerank_score"] if reranked else 0.0
        tier = self.tiers.classify_score(top_score)

        stats = {
            "method": ("HyDE + " if hyde_used else "") + "Hybrid (RRF) + Rerank",
            "route": route,
            "hyde_used": hyde_used,
            "num_candidates_evaluated": len(candidates),
            "hyde_latency_ms": hyde_latency,
            "hybrid_latency_ms": hybrid_latency,
            "rerank_latency_ms": rerank_latency,
            "top_score": float(top_score),
            "tier": tier.value,
            "high_threshold": self.tiers.high_threshold,
            "low_threshold": self.tiers.low_threshold,
            "fallback_triggered": tier is ConfidenceTier.LOW,
        }
        if verbose:
            logger.info(f"[Retrieve] {stats}")

        if tier is ConfidenceTier.LOW:
            logger.warning(
                f"Low confidence retrieval: top_score={top_score:.3f} < low_threshold={self.tiers.low_threshold}."
            )
            return {
                "query": query, "hyde_generated": dense_text, "final_chunks": [],
                "confidence": tier.value,
                "message": "Low confidence retrieval. Recommend query reformulation.",
                "retrieval_stats": stats,
            }
        return {
            "query": query, "hyde_generated": dense_text, "final_chunks": reranked,
            "confidence": tier.value, "retrieval_stats": stats,
        }
