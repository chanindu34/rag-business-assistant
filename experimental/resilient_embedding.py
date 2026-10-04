"""
Resilient Embedding: Batching + exponential backoff + local fallback.

Solves: API quota exhaustion during ingest stalls the pipeline.
Impact: Batch requests, retry intelligently, fall back to local model if needed.
"""

import logging
import time
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = logging.getLogger(__name__)


class ResilientEmbedder:
    """
    Embed texts with batching, exponential backoff, and local model fallback.
    """

    def __init__(self, api_embedder, fallback_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            api_embedder: Gemini embedder instance (from ingest.py)
            fallback_model: HuggingFace model name for local fallback
        """
        self.api_embedder = api_embedder
        self.fallback_model = fallback_model
        self.use_fallback = False
        self.local_model = None

        # Try to load fallback model once
        self._init_fallback()

    def _init_fallback(self):
        """Load local embedding model on init."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[Embedder] Loading fallback model: {self.fallback_model}")
            self.local_model = SentenceTransformer(self.fallback_model)
            logger.info(f"[Embedder] Fallback model ready: {self.fallback_model}")
        except Exception as e:
            logger.warning(f"[Embedder] Could not load fallback model: {e}. Will only use API.")
            self.local_model = None

    def _is_quota_error(self, exception: Exception) -> bool:
        """Check if exception is a quota/rate limit error."""
        error_str = str(exception).lower()
        return any(keyword in error_str for keyword in [
            "429", "resource_exhausted", "quota", "rate_limit", "too_many_requests"
        ])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception(lambda e: not str(e).lower().count("resource_exhausted") > 0),
        reraise=True
    )
    def _embed_batch_with_api(self, texts: List[str]) -> List:
        """
        Embed batch via API with exponential backoff.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        logger.info(f"[Embedder] Embedding batch of {len(texts)} via API...")
        embeddings = []

        for text in texts:
            try:
                emb = self.api_embedder.embed_content(text)
                embeddings.append(emb)
                time.sleep(0.05)  # Small delay between calls
            except Exception as e:
                if self._is_quota_error(e):
                    raise  # Let retry handler catch quota errors
                else:
                    raise  # Re-raise other errors

        return embeddings

    def embed_batch(self, texts: List[str], batch_size: int = 50) -> List:
        """
        Embed texts in batches with fallback logic.

        Strategy:
        1. Try API in batches with exponential backoff
        2. On quota error → switch to local fallback for remaining
        3. Log all decisions

        Args:
            texts: List of text strings to embed
            batch_size: How many texts per batch

        Returns:
            List of embedding vectors (same length as texts)
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            logger.info(f"[Embedder] Batch {batch_num + 1}/{total_batches} ({len(batch)} texts)")

            if self.use_fallback:
                # Already switched to fallback
                logger.info(f"[Embedder] Using local fallback for this batch")
                embeddings = self.local_model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings)

            else:
                # Try API with backoff
                try:
                    embeddings = self._embed_batch_with_api(batch)
                    all_embeddings.extend(embeddings)

                except Exception as e:
                    if self._is_quota_error(e):
                        logger.warning(f"[Embedder] API quota exhausted. Switching to local fallback.")

                        if not self.local_model:
                            logger.error("[Embedder] Local fallback not available. Failing.")
                            raise RuntimeError("API quota hit but no fallback model loaded.")

                        self.use_fallback = True

                        # Embed current batch with fallback
                        embeddings = self.local_model.encode(batch, show_progress_bar=False)
                        all_embeddings.extend(embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings)
                    else:
                        raise

        logger.info(
            f"[Embedder] Completed {len(texts)} embeddings. "
            f"API used: {not self.use_fallback}. Fallback used: {self.use_fallback}"
        )

        return all_embeddings
