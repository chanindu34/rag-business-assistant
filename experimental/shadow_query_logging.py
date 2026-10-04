"""
Shadow Query Logging: Dead-Letter Queue for Low-Confidence Queries

Enterprise insight: Every query where reranker confidence < threshold tells you
exactly what your knowledge base FAILS to answer. This is gold.

Use case: After Oct 31 report launch, log shows "50 queries about revised dividend",
"20 about Q2 guidance". Engineering team sees: knowledge base is missing key sections.
Next sprint: ingest those documents first.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ShadowQuery:
    """A query that fell below confidence threshold."""
    query: str
    confidence_score: float
    confidence_level: str  # "low", "medium", "high"
    retrieved_chunks: List[str]
    reranker_scores: List[float]
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ShadowQueryLogger:
    """
    Log failed queries to a dead-letter queue.

    Queries with confidence < 0.35 indicate knowledge base gaps.
    Periodically review these to improve training data.
    """

    def __init__(
        self,
        log_file: str = "shadow_queries.jsonl",
        confidence_threshold: float = 0.35,
    ):
        """
        Args:
            log_file: Path to write low-confidence queries (JSONL format)
            confidence_threshold: Log queries below this confidence
        """
        self.log_file = Path(log_file)
        self.confidence_threshold = confidence_threshold
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Create log file if it doesn't exist."""
        if not self.log_file.exists():
            self.log_file.touch()

    def log_query(
        self,
        query: str,
        confidence_score: float,
        confidence_level: str,
        retrieved_chunks: List[str],
        reranker_scores: List[float],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Log a query if it falls below threshold.

        Returns:
            True if logged, False if above threshold
        """
        if confidence_score >= self.confidence_threshold:
            return False  # Above threshold, don't log

        shadow = ShadowQuery(
            query=query,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            retrieved_chunks=retrieved_chunks,
            reranker_scores=reranker_scores,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            session_id=session_id,
        )

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(shadow.to_dict()) + '\n')

        logger.info(
            f"Shadow query logged: '{query[:50]}...' "
            f"(confidence={confidence_score:.3f})"
        )
        return True

    def read_shadow_queries(
        self,
        limit: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ShadowQuery]:
        """
        Read shadow queries from log.

        Args:
            limit: Max queries to read
            min_confidence: Only return queries below this score

        Returns:
            List of ShadowQuery objects
        """
        queries = []
        with open(self.log_file) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                shadow = ShadowQuery(**data)

                if min_confidence and shadow.confidence_score >= min_confidence:
                    continue

                queries.append(shadow)

                if limit and len(queries) >= limit:
                    break

        return queries

    def get_summary_stats(self) -> Dict:
        """
        Analyze shadow queries to find knowledge gaps.

        Returns:
            {
                "total_logged": int,
                "avg_confidence": float,
                "top_failed_queries": List[str],
                "failed_query_count": int,
            }
        """
        queries = self.read_shadow_queries()

        if not queries:
            return {
                "total_logged": 0,
                "avg_confidence": 0.0,
                "top_failed_queries": [],
                "failed_query_count": 0,
            }

        # Sort by confidence
        sorted_queries = sorted(
            queries,
            key=lambda q: q.confidence_score
        )

        # Top 5 lowest-confidence (worst performing)
        top_worst = sorted_queries[:5]

        avg_confidence = sum(q.confidence_score for q in queries) / len(queries)

        return {
            "total_logged": len(queries),
            "avg_confidence": round(avg_confidence, 3),
            "top_failed_queries": [q.query for q in top_worst],
            "failed_query_count": len(sorted_queries),
        }

    def export_for_review(self, output_file: str = "shadow_queries_export.json"):
        """
        Export shadow queries for engineering team review.
        Identifies priority documents to ingest next.
        """
        queries = self.read_shadow_queries()

        # Group by query theme (simple: just export with analysis)
        summary = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_queries": len(queries),
            "avg_confidence": round(
                sum(q.confidence_score for q in queries) / len(queries)
                if queries else 0.0,
                3
            ),
            "queries": [q.to_dict() for q in queries[:50]],  # Last 50
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported {len(queries)} shadow queries to {output_file}")
        return output_file


class ConfidenceThresholdOptimizer:
    """
    Find the optimal confidence threshold.

    Too high threshold: Everything logged, too noisy.
    Too low threshold: Miss real issues.

    Analyze: At what confidence score does quality drop visibly?
    """

    @staticmethod
    def analyze_confidence_distribution(
        queries: List[ShadowQuery],
        bins: int = 10,
    ) -> Dict:
        """
        Histogram of confidence scores.
        Shows where queries cluster.
        """
        if not queries:
            return {}

        scores = [q.confidence_score for q in queries]

        # Bin into 10 buckets [0.0-0.1], [0.1-0.2], ..., [0.9-1.0]
        bins_dict = {
            f"{i*0.1:.1f}-{(i+1)*0.1:.1f}": 0
            for i in range(bins)
        }

        for score in scores:
            bin_idx = min(int(score * bins), bins - 1)
            bin_key = f"{bin_idx*0.1:.1f}-{(bin_idx+1)*0.1:.1f}"
            bins_dict[bin_key] += 1

        return {
            "confidence_distribution": bins_dict,
            "mean_confidence": round(sum(scores) / len(scores), 3),
            "min_confidence": round(min(scores), 3),
            "max_confidence": round(max(scores), 3),
        }

    @staticmethod
    def recommend_threshold(queries: List[ShadowQuery]) -> float:
        """
        Recommend confidence threshold based on gap analysis.

        Find the "elbow" in the distribution where quality drops.
        """
        if not queries:
            return 0.35

        scores = sorted([q.confidence_score for q in queries])

        # Simple heuristic: threshold at 25th percentile
        # = capture the worst 25% of queries
        threshold = scores[len(scores) // 4]

        return round(threshold, 2)
