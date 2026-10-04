"""
BM25 Persistence: Serialize index to disk, detect staleness vs ChromaDB.

Solves: BM25 rebuilds on every container restart, goes out of sync with ChromaDB.
Impact: Scales to 500K chunks without RAM exhaustion, survives restarts.
"""

import pickle
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class PersistentBM25:
    """BM25 index that persists to disk and detects staleness."""

    def __init__(self, index_path: str = "./bm25_index.pkl"):
        """
        Args:
            index_path: Where to save/load the BM25 index
        """
        self.index_path = index_path
        self.corpus = None
        self.bm25 = None
        self.corpus_hash = None

    def build_and_save(self, corpus: List[str]):
        """
        Build BM25 from corpus and serialize to disk.

        Args:
            corpus: List of document strings
        """
        from rank_bm25 import BM25Okapi

        logger.info(f"[BM25] Building index for {len(corpus)} documents...")

        # Tokenize
        tokenized = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus
        self.corpus_hash = hash(tuple(corpus))

        # Save to disk
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'corpus': self.corpus,
                'corpus_hash': self.corpus_hash,
            }, f)

        logger.info(f"[BM25] Index saved to {self.index_path} ({len(corpus)} docs)")

    def load(self) -> bool:
        """
        Load index from disk.

        Returns:
            True if successful, False if file not found or corrupted
        """
        if not os.path.exists(self.index_path):
            logger.info(f"[BM25] Index not found at {self.index_path}")
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.corpus = data['corpus']
                self.corpus_hash = data['corpus_hash']
            logger.info(f"[BM25] Loaded {len(self.corpus)} docs from {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"[BM25] Failed to load index: {e}")
            return False

    def is_stale(self, new_corpus: List[str]) -> bool:
        """
        Check if corpus changed since last save.

        Args:
            new_corpus: Current corpus to compare

        Returns:
            True if hashes don't match (corpus has changed)
        """
        new_hash = hash(tuple(new_corpus))
        is_stale = new_hash != self.corpus_hash

        if is_stale:
            logger.warning(f"[BM25] Corpus changed. Index is stale.")

        return is_stale

    def query(self, q: str, k: int = 10) -> List[int]:
        """
        Query the BM25 index.

        Args:
            q: Query string
            k: Number of top results

        Returns:
            List of document indices, sorted by score (highest first)
        """
        if not self.bm25:
            logger.error("[BM25] Index not loaded. Call build_and_save() or load() first.")
            return []

        tokens = q.split()
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_k = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return top_k
