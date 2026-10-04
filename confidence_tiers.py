"""
Dynamic Confidence Tiers: Replace binary pass/fail with tri-tier routing.

Tier 1 (S >= 0.65): High Confidence → Direct answer
Tier 2 (0.35 <= S < 0.65): Ambiguity Zone → Self-correction loop (query rewrite)
Tier 3 (S < 0.35): Out of Domain → Graceful exit with "I don't know"

This prevents hallucinations in the gray zone by triggering query refinement.
"""

import logging
from typing import Dict, List, Literal
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceTier(Enum):
    """Confidence tiers based on reranker score."""
    HIGH = "high"           # S >= 0.65
    AMBIGUOUS = "ambiguous" # 0.35 <= S < 0.65
    LOW = "low"             # S < 0.35


class ConfidenceTierRouter:
    """
    Route queries based on reranker confidence score into three tiers.
    """

    def __init__(
        self,
        high_threshold: float = 0.65,
        low_threshold: float = 0.35,
    ):
        """
        Args:
            high_threshold: Score >= this → HIGH confidence
            low_threshold: Score < this → LOW confidence
                           Between them → AMBIGUOUS
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def classify_score(self, score: float) -> ConfidenceTier:
        """Classify a reranker score into a tier."""
        if score >= self.high_threshold:
            return ConfidenceTier.HIGH
        elif score >= self.low_threshold:
            return ConfidenceTier.AMBIGUOUS
        else:
            return ConfidenceTier.LOW

    def route(self, query: str, top_score: float, chunks: List[str]) -> Dict:
        """
        Route based on confidence tier.

        Returns:
            {
                "tier": ConfidenceTier,
                "action": "answer" | "retry" | "fail",
                "message": str,
                "chunks": List[str] (for answer tier)
            }
        """
        tier = self.classify_score(top_score)

        if tier == ConfidenceTier.HIGH:
            # Confident: Use chunks directly
            logger.info(f"[Confidence] Score {top_score:.3f} → HIGH tier → Direct answer")
            return {
                "tier": tier,
                "action": "answer",
                "message": None,
                "chunks": chunks,
            }

        elif tier == ConfidenceTier.AMBIGUOUS:
            # Ambiguous: Suggest query rewrite
            logger.info(f"[Confidence] Score {top_score:.3f} → AMBIGUOUS tier → Query rewrite")
            return {
                "tier": tier,
                "action": "retry",
                "message": "Confidence is moderate. Trying query reformulation...",
                "chunks": chunks,  # Can use for secondary attempt
                "suggested_rewrites": self._suggest_rewrites(query),
            }

        else:  # LOW
            # Out of domain: Fail gracefully
            logger.info(f"[Confidence] Score {top_score:.3f} → LOW tier → Graceful exit")
            return {
                "tier": tier,
                "action": "fail",
                "message": "I do not have sufficient internal documentation to answer this question with confidence.",
                "chunks": [],
            }

    def _suggest_rewrites(self, query: str) -> List[str]:
        """
        Suggest alternative query phrasings for the ambiguous tier.
        (In production, this could be LLM-driven; for now, heuristic-based.)
        """
        rewrites = []

        # Strategy 1: Remove conjunctions and make more specific
        if " and " in query.lower():
            rewrites.append(query.split(" and ")[0])

        # Strategy 2: Add domain context
        rewrites.append(f"What information about {query} is in the annual report?")

        # Strategy 3: Simplify to core terms
        words = query.split()
        if len(words) > 5:
            rewrites.append(" ".join(words[:5]))

        return rewrites[:2]  # Return top 2 suggestions


class ConfidenceTierTelemetry:
    """Track confidence tier distribution and outcomes."""

    def __init__(self):
        self.high_count = 0
        self.ambiguous_count = 0
        self.low_count = 0
        self.retry_attempts = 0
        self.retry_success_count = 0

    def record_tier(self, tier: ConfidenceTier):
        """Record a query classified into a tier."""
        if tier == ConfidenceTier.HIGH:
            self.high_count += 1
        elif tier == ConfidenceTier.AMBIGUOUS:
            self.ambiguous_count += 1
        else:
            self.low_count += 1

    def record_retry(self, success: bool):
        """Record a retry attempt and outcome."""
        self.retry_attempts += 1
        if success:
            self.retry_success_count += 1

    def report(self) -> dict:
        """Return telemetry summary."""
        total = self.high_count + self.ambiguous_count + self.low_count
        return {
            "total_queries": total,
            "high_confidence": self.high_count,
            "ambiguous": self.ambiguous_count,
            "low_confidence": self.low_count,
            "ambiguous_percent": (self.ambiguous_count / total * 100) if total > 0 else 0,
            "retry_attempts": self.retry_attempts,
            "retry_success_rate": (
                self.retry_success_count / self.retry_attempts
                if self.retry_attempts > 0
                else 0
            ),
        }
