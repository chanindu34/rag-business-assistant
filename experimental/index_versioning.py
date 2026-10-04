"""
Index Versioning & Embedding Drift Detection

Enterprise production RAG must handle:
1. Document version lifecycle (new reports override old ones)
2. Embedding model updates (old vectors become invalid with new models)
3. Blue-green re-indexing (zero-downtime updates)
4. Drift detection (monitor when query performance degrades)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndexVersion:
    """Metadata for a specific index version."""
    version_id: str
    embedding_model: str
    chunk_size: int
    created_at: str
    document_source: str  # e.g., "Annual_Report_2025_26"
    total_chunks: int
    is_active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'IndexVersion':
        return IndexVersion(**data)


class IndexVersionManager:
    """
    Manage multiple index versions.

    Scenario: Your Annual Report 2026/27 drops. Create new index version,
    run benchmarks, switch traffic atomically. Old version stays for rollback.
    """

    def __init__(self, versions_dir: str = ".index_versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(exist_ok=True)
        self.metadata_file = self.versions_dir / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict[str, IndexVersion]:
        """Load all known index versions from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                return {
                    vid: IndexVersion.from_dict(v)
                    for vid, v in data.items()
                }
        return {}

    def _save_versions(self):
        """Persist index versions to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(
                {vid: v.to_dict() for vid, v in self.versions.items()},
                f,
                indent=2
            )

    def register_version(
        self,
        version_id: str,
        embedding_model: str,
        chunk_size: int,
        document_source: str,
        total_chunks: int,
    ) -> IndexVersion:
        """Register a new index version (after ingestion)."""
        version = IndexVersion(
            version_id=version_id,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            created_at=datetime.utcnow().isoformat(),
            document_source=document_source,
            total_chunks=total_chunks,
            is_active=False  # Don't activate until benchmarked
        )
        self.versions[version_id] = version
        self._save_versions()
        logger.info(f"Registered index version: {version_id}")
        return version

    def activate_version(self, version_id: str) -> bool:
        """Atomically switch to a new version (after passing quality gates)."""
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return False

        # Deactivate all others
        for v in self.versions.values():
            v.is_active = False

        # Activate this one
        self.versions[version_id].is_active = True
        self._save_versions()
        logger.info(f"Activated index version: {version_id}")
        return True

    def get_active_version(self) -> Optional[IndexVersion]:
        """Get the currently active index version."""
        for v in self.versions.values():
            if v.is_active:
                return v
        return None

    def get_version_history(self) -> List[IndexVersion]:
        """Get all versions, sorted by creation time."""
        return sorted(
            self.versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )


class EmbeddingDriftDetector:
    """
    Detect when embedding model changes invalidate old vectors.

    Strategy: Sample queries periodically. If new embeddings diverge
    significantly from old ones, flag drift.
    """

    def __init__(self, drift_threshold: float = 0.15):
        """
        Args:
            drift_threshold: Cosine distance threshold to flag drift.
                If mean distance > threshold, embeddings have drifted.
        """
        self.drift_threshold = drift_threshold
        self.drift_log = []

    def detect_drift(
        self,
        old_embeddings: List[List[float]],
        new_embeddings: List[List[float]],
        sample_size: int = 100
    ) -> Dict:
        """
        Compare old vs new embeddings on a sample.

        Returns:
            {
                "has_drifted": bool,
                "mean_cosine_distance": float,
                "max_distance": float,
                "samples_tested": int
            }
        """
        old_emb = np.array(old_embeddings)
        new_emb = np.array(new_embeddings)

        # Sample both
        n = min(len(old_emb), len(new_emb), sample_size)
        indices = np.random.choice(len(old_emb), n, replace=False)

        # Compute cosine distance for each pair
        distances = []
        for i in indices:
            old_vec = old_emb[i]
            new_vec = new_emb[i]

            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(old_vec, new_vec) / (
                np.linalg.norm(old_vec) * np.linalg.norm(new_vec)
            )
            distance = 1 - similarity
            distances.append(distance)

        distances = np.array(distances)
        mean_distance = float(np.mean(distances))
        max_distance = float(np.max(distances))

        has_drifted = mean_distance > self.drift_threshold

        result = {
            "has_drifted": has_drifted,
            "mean_cosine_distance": mean_distance,
            "max_distance": max_distance,
            "samples_tested": n,
            "threshold": self.drift_threshold,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.drift_log.append(result)

        if has_drifted:
            logger.warning(
                f"EMBEDDING DRIFT DETECTED: mean_distance={mean_distance:.4f} "
                f"(threshold={self.drift_threshold}). Recommend re-indexing."
            )

        return result


class DocumentLifecycleTracker:
    """
    Track document chunk metadata: version, effective date, obsolescence.

    Chunks inherit from source document. When Annual_Report_2026_27
    arrives, mark 2025_26 chunks as superseded.
    """

    def __init__(self):
        self.chunk_metadata = {}  # chunk_id -> metadata

    def register_chunk(
        self,
        chunk_id: str,
        source_document: str,
        doc_version: str,
        effective_date: str,
        is_active: bool = True,
    ):
        """Register a chunk's lifecycle metadata."""
        self.chunk_metadata[chunk_id] = {
            "source_document": source_document,
            "doc_version": doc_version,
            "effective_date": effective_date,
            "is_active": is_active,
            "created_at": datetime.utcnow().isoformat(),
        }

    def supersede_document_version(
        self,
        source_document: str,
        old_version: str,
    ):
        """Mark all chunks from old version as inactive."""
        count = 0
        for chunk_id, metadata in self.chunk_metadata.items():
            if (metadata["source_document"] == source_document and
                metadata["doc_version"] == old_version):
                metadata["is_active"] = False
                count += 1
        logger.info(
            f"Marked {count} chunks from {source_document}:{old_version} as inactive"
        )

    def get_active_chunks(self) -> List[str]:
        """Get IDs of all active chunks (current versions)."""
        return [
            cid for cid, meta in self.chunk_metadata.items()
            if meta["is_active"]
        ]

    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict]:
        """Get lifecycle metadata for a chunk."""
        return self.chunk_metadata.get(chunk_id)
