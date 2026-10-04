"""
Production RAG Retriever: HyDE + Hybrid Search + Cross-Encoder Reranking

Three-layer retrieval pipeline:
1. HyDE (Hypothetical Document Embeddings): Generate ideal answer, embed that
2. Hybrid Search: Combine BM25 (keyword) + Vector (semantic) using RRF
3. Reranking: Score top 20 with cross-encoder, return top 3
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class HyDETransformer:
    """
    Hypothetical Document Embeddings.
    Generate an ideal answer to the query, then use that for embedding.
    Why? Better semantic representation than raw query.
    """

    def __init__(self, llm_client, generation_model: str, generate_fn=None, enabled: bool = True):
        self.llm_client = llm_client
        self.generation_model = generation_model
        # generate_fn(prompt) -> str. Lets the app route HyDE through the
        # quota-aware gateway (fallback models, cache) instead of a raw call.
        self.generate_fn = generate_fn
        self.enabled = enabled

    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate what an ideal answer to this question would look like."""
        prompt = f"""Generate a plausible, detailed answer to this question.
Be specific, as if answering based on a document.
Only generate the answer, no preamble.

Question: {query}

Answer:"""

        if not self.enabled:
            return query
        try:
            if self.generate_fn is not None:
                text = (self.generate_fn(prompt) or "").strip()
                return text or query
            response = self.llm_client.models.generate_content(
                model=self.generation_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.7,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                logger.warning("HyDE returned empty text. Using original query.")
                return query
            return text
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Using original query.")
            return query

    def transform(self, query: str) -> Dict[str, str]:
        """Return both original query and hypothetical answer."""
        hypo = self.generate_hypothetical_answer(query)
        return {
            "original_query": query,
            "hypothetical_answer": hypo
        }


class HybridRetriever:
    """
    Combine sparse (BM25) and dense (vector) retrieval using Reciprocal Rank Fusion.

    - BM25: Catches exact matches (SKUs, acronyms, domain terms)
    - Vector: Catches semantic meaning
    - Together: Best of both worlds
    """

    def __init__(self, chunks: List[str], embeddings: List[List[float]]):
        """
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors (already computed)
        """
        self.chunks = chunks
        self.embeddings = np.array(embeddings)

        # Build BM25 index
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

        logger.info(f"Initialized HybridRetriever with {len(chunks)} chunks")

    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 20,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> Tuple[List[str], List[float]]:
        """
        Hybrid retrieval: combine BM25 + vector search.

        Args:
            query_text: Original user query (for BM25)
            query_embedding: Vector embedding of HyDE hypothetical answer
            k: Number of candidates to return
            bm25_weight: Weight for BM25 scores (0-1)
            vector_weight: Weight for vector scores (0-1)

        Returns:
            (top_k_chunks, combined_scores)
        """
        # 1. BM25 scores (sparse retrieval)
        tokenized_query = query_text.split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float32)

        # 2. Vector scores (dense retrieval)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        vector_scores = np.dot(self.embeddings, query_embedding)

        # 3. Normalize both to [0, 1]
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        bm25_norm = (
            (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-10)
            if bm25_max > bm25_min
            else np.zeros_like(bm25_scores)
        )

        vector_min, vector_max = vector_scores.min(), vector_scores.max()
        vector_norm = (
            (vector_scores - vector_min) / (vector_max - vector_min + 1e-10)
            if vector_max > vector_min
            else np.zeros_like(vector_scores)
        )

        # 4. Weighted combination
        combined = (bm25_weight * bm25_norm) + (vector_weight * vector_norm)

        # 5. Get top K indices
        top_indices = np.argsort(combined)[::-1][:k]

        top_chunks = [self.chunks[i] for i in top_indices]
        top_scores = combined[top_indices]

        return top_chunks, top_scores


class RerankerFilter:
    """
    Cross-Encoder Reranking.

    Takes top 20 candidates, scores them semantically, returns top 3.
    Ensures final chunks are actually relevant.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """Load a cross-encoder reranker."""
        try:
            self.reranker = CrossEncoder(model_name)
            logger.info(f"Loaded reranker: {model_name}")
        except Exception as e:
            # Fail loudly. A missing reranker used to fall back to score=1.0,
            # which silently marked every answer as HIGH confidence.
            raise RuntimeError(f"Failed to load reranker {model_name}: {e}") from e

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 3,
    ) -> Tuple[List[Dict], float]:
        """
        Score candidates and return top K.

        Args:
            query: User query
            candidates: List of candidate chunks (typically top 20 from retriever)
            top_k: Number to return (typically 3)

        Returns:
            (reranked_results, latency_ms)
        """

        start = time.time()

        try:
            # Score each candidate against query
            pairs = [[query, candidate] for candidate in candidates]
            scores = self.reranker.predict(pairs)

            latency = (time.time() - start) * 1000  # ms

            # Sort by score (highest first)
            sorted_indices = np.argsort(scores)[::-1]

            # Return top K with metadata
            reranked = [
                {
                    "chunk": candidates[i],
                    "rerank_score": float(scores[i]),
                    "rank": j,
                }
                for j, i in enumerate(sorted_indices[:top_k])
            ]

            return reranked, latency

        except Exception as e:
            # Score 0.0, not 1.0: an unscored answer must land in the LOW tier.
            logger.error(f"Reranking failed: {e}. Marking results as unscored.")
            return [{"chunk": c, "rerank_score": 0.0, "rank": i}
                    for i, c in enumerate(candidates[:top_k])], 0.0


class ProductionRAG:
    """
    Complete production RAG pipeline:
    1. HyDE: Generate hypothetical answer from query
    2. Hybrid: Retrieve top 20 (BM25 + vector)
    3. Rerank: Score top 20, return top 3
    4. Confidence Guardrails: Fail gracefully if confidence too low
    """

    def __init__(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        embedding_model: str,
        generation_model: str,
        llm_client,
        reranker_model: str = "BAAI/bge-reranker-base",
        confidence_threshold: float = 0.4,
        hyde_generate_fn=None,
        use_hyde: bool = True,
    ):
        """
        Args:
            chunks: List of text chunks
            embeddings: Pre-computed embeddings for chunks
            embedding_model: Embedding model name (for getting query embedding)
            generation_model: Generation model name (for HyDE)
            llm_client: Genai client
            reranker_model: Cross-encoder model name
            confidence_threshold: Minimum reranker score to return results (0-1)
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.confidence_threshold = confidence_threshold

        self.hyde = HyDETransformer(llm_client, generation_model, generate_fn=hyde_generate_fn, enabled=use_hyde)
        self.retriever = HybridRetriever(chunks, embeddings)
        self.reranker = RerankerFilter(reranker_model)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        verbose: bool = False,
    ) -> Dict:
        """
        Full three-layer retrieval pipeline with confidence guardrails.

        Returns:
            {
                "query": original query,
                "hyde_generated": hypothetical answer,
                "final_chunks": list of reranked chunks with scores (empty if low confidence),
                "confidence": "high" or "low",
                "retrieval_stats": {
                    "num_candidates_evaluated": number evaluated,
                    "rerank_latency_ms": latency,
                    "top_score": confidence score of top result,
                    "threshold": confidence threshold used,
                    "fallback_triggered": whether low confidence triggered fallback,
                    "method": "HyDE + Hybrid + Rerank"
                }
            }
        """
        if verbose:
            print(f"\n[Retrieval Pipeline] Query: {query}")

        # Step 1: HyDE transformation
        start_hyde = time.time()
        transformed = self.hyde.transform(query)
        original_query = transformed["original_query"]
        hypothetical = transformed["hypothetical_answer"]
        hyde_latency = (time.time() - start_hyde) * 1000

        if verbose:
            print(f"[HyDE] Generated (latency {hyde_latency:.0f}ms):")
            print(f"  {hypothetical[:100]}...")

        # Step 2: Hybrid retrieval (top 20)
        start_hybrid = time.time()

        # Get embedding of hypothetical answer
        query_embedding = self.llm_client.models.embed_content(
            model=self.embedding_model,
            contents=hypothetical
        ).embeddings[0].values

        candidates, hybrid_scores = self.retriever.retrieve(
            original_query,
            query_embedding,
            k=20,
        )
        hybrid_latency = (time.time() - start_hybrid) * 1000

        if verbose:
            print(f"[Hybrid] Retrieved {len(candidates)} candidates (latency {hybrid_latency:.0f}ms)")

        # Step 3: Rerank (top K)
        reranked, rerank_latency = self.reranker.rerank(
            original_query,
            candidates,
            top_k=top_k,
        )

        if verbose:
            print(f"[Rerank] Scored and reranked (latency {rerank_latency:.0f}ms)")
            for r in reranked:
                print(f"  Rank {r['rank']}: score={r['rerank_score']:.3f}")

        # Step 4: Confidence Guardrails
        top_score = reranked[0]["rerank_score"] if reranked else 0.0
        is_confident = top_score >= self.confidence_threshold

        if verbose:
            print(f"[Confidence] Top score: {top_score:.3f}, threshold: {self.confidence_threshold}")
            print(f"[Confidence] Result: {'HIGH' if is_confident else 'LOW'}")

        if not is_confident:
            logger.warning(
                f"Low confidence retrieval: top_score={top_score:.3f} < "
                f"threshold={self.confidence_threshold}. Recommending fallback."
            )
            return {
                "query": original_query,
                "hyde_generated": hypothetical,
                "final_chunks": [],
                "confidence": "low",
                "message": "Low confidence retrieval. Recommend query reformulation or external search.",
                "retrieval_stats": {
                    "num_candidates_evaluated": len(candidates),
                    "rerank_latency_ms": rerank_latency,
                    "method": "HyDE + Hybrid + Rerank",
                    "hyde_latency_ms": hyde_latency,
                    "hybrid_latency_ms": hybrid_latency,
                    "top_score": float(top_score),
                    "threshold": self.confidence_threshold,
                    "fallback_triggered": True,
                }
            }

        return {
            "query": original_query,
            "hyde_generated": hypothetical,
            "final_chunks": reranked,
            "confidence": "high",
            "retrieval_stats": {
                "num_candidates_evaluated": len(candidates),
                "rerank_latency_ms": rerank_latency,
                "method": "HyDE + Hybrid + Rerank",
                "hyde_latency_ms": hyde_latency,
                "hybrid_latency_ms": hybrid_latency,
                "top_score": float(top_score),
                "threshold": self.confidence_threshold,
                "fallback_triggered": False,
            }
        }
