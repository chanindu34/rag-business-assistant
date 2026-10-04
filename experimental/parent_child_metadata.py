"""
Parent-Child Chunking: Hierarchical retrieval for precision + context.

Strategy:
- Child chunks: Small (100-200 tokens) for precise vector search + BM25
- Parent chunks: Large (800-1000 tokens) for LLM context
- Metadata: Store parent_id on each child, fetch parent when needed

Result: Search returns precise child, but LLM gets rich parent context.
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class ParentChildChunker:
    """
    Split documents into child chunks (small, searchable) and parent chunks (large, contextual).
    """

    def __init__(
        self,
        child_chunk_size: int = 150,
        child_overlap: int = 20,
        parent_chunk_size: int = 800,
        parent_overlap: int = 100,
    ):
        """
        Args:
            child_chunk_size: Tokens per child chunk (for search)
            child_overlap: Overlap between child chunks
            parent_chunk_size: Tokens per parent chunk (for LLM)
            parent_overlap: Overlap between parent chunks
        """
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap

    def split_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Split text into parent and child chunks.

        Returns:
            (parent_chunks, child_chunks)
        """
        # First, create parent chunks (larger)
        parent_chunks = self._create_chunks(
            text,
            chunk_size=self.parent_chunk_size,
            overlap=self.parent_overlap,
        )

        # Then, create child chunks (smaller, more granular)
        child_chunks = self._create_chunks(
            text,
            chunk_size=self.child_chunk_size,
            overlap=self.child_overlap,
        )

        logger.info(f"[Parent-Child] Split into {len(parent_chunks)} parents and {len(child_chunks)} children")

        return parent_chunks, child_chunks

    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Create overlapping chunks of specified size."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def create_metadata(
        self, parent_chunks: List[str], child_chunks: List[str]
    ) -> Dict[int, Dict]:
        """
        Create metadata mapping: child_idx → parent info.

        Returns:
            {
                child_idx: {
                    "parent_id": parent_idx,
                    "parent_text": parent_chunk_text,
                    "child_text": child_chunk_text,
                }
            }
        """
        metadata = {}

        # Map each child to its closest parent
        for child_idx, child in enumerate(child_chunks):
            # Find parent with highest overlap
            best_parent_idx = 0
            best_overlap = 0

            child_words = set(child.split())

            for parent_idx, parent in enumerate(parent_chunks):
                parent_words = set(parent.split())
                overlap = len(child_words & parent_words)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_parent_idx = parent_idx

            metadata[child_idx] = {
                "parent_id": best_parent_idx,
                "parent_text": parent_chunks[best_parent_idx],
                "child_text": child,
            }

        logger.info(f"[Parent-Child] Created metadata for {len(metadata)} children")
        return metadata


class ParentChildRetriever:
    """
    Retrieve child chunks (for search), but return parent chunks (for LLM context).
    """

    def __init__(self, metadata: Dict[int, Dict]):
        """
        Args:
            metadata: Output from ParentChildChunker.create_metadata()
        """
        self.metadata = metadata

    def get_parent_for_child(self, child_idx: int) -> str:
        """Given a child chunk index, return the parent chunk text."""
        if child_idx not in self.metadata:
            logger.warning(f"[Parent-Child] Child idx {child_idx} not in metadata")
            return None

        return self.metadata[child_idx]["parent_text"]

    def enrich_retrieved_chunks(self, child_indices: List[int]) -> List[str]:
        """
        Given retrieved child indices, return parent chunks for LLM.

        Deduplicates: if multiple children map to same parent, return parent once.
        """
        parent_ids = set()
        parent_chunks = []

        for child_idx in child_indices:
            if child_idx in self.metadata:
                parent_id = self.metadata[child_idx]["parent_id"]
                if parent_id not in parent_ids:
                    parent_ids.add(parent_id)
                    parent_text = self.metadata[child_idx]["parent_text"]
                    parent_chunks.append(parent_text)

        logger.info(f"[Parent-Child] Enriched {len(child_indices)} children → {len(parent_chunks)} unique parents")
        return parent_chunks


# Integration point: Modify ingest.py to use this
def enhance_ingest_with_parent_child(
    chunks: List[str],
    chunk_size: int = 500,
    parent_chunk_size: int = 800,
) -> Tuple[List[str], List[str], Dict]:
    """
    Wrapper to use in ingest.py.

    Takes existing chunks, creates parent-child hierarchy.

    Returns:
        (child_chunks_for_search, parent_chunks_for_context, metadata)
    """
    text = " ".join(chunks)  # Reconstruct from chunks

    chunker = ParentChildChunker(
        child_chunk_size=chunk_size,
        child_overlap=50,
        parent_chunk_size=parent_chunk_size,
        parent_overlap=100,
    )

    parent_chunks, child_chunks = chunker.split_text(text)
    metadata = chunker.create_metadata(parent_chunks, child_chunks)

    return child_chunks, parent_chunks, metadata
