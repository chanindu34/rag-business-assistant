"""
Semantic Router: decide whether a query is worth a HyDE LLM call.

- Explanatory questions ("why", "explain", "strategy", "risks", "outlook")
  benefit from HyDE: the drafted answer is closer to report prose than the
  short question is.
- Fact lookups (numbers, years, acronyms like EBITDA/PBT, "how much",
  "how many", named metrics) are better served by exact keyword matching,
  so HyDE is skipped. This saves one LLM call per lookup.
- Explanatory signals win over lookup signals: "Why did EBITDA rise 75%?"
  still runs HyDE.
- Anything unclear defaults to running HyDE (quality over cost).
"""

import logging
import re
from typing import Literal

logger = logging.getLogger(__name__)

Route = Literal["skip_hyde", "run_hyde"]

_CONCEPTUAL = re.compile(
    r"\b(why|explain\w*|describe|discuss|analy[sz]\w*|compare|comparison|relationship|"
    r"impacts?|effects?|implications?|strateg\w*|approach|outlook|risks?|challenges?|"
    r"opportunit\w+|drivers?|overview|summari[sz]\w*|priorit\w+|plans?|vision|"
    r"how (?:does|did|do|is|are|will|has|have|can|could|should))\b",
    re.IGNORECASE,
)
_LOOKUP_PHRASE = re.compile(r"\bhow (?:much|many)\b|\bwhat (?:is|was|were|are) the (?:total|number|amount|value)\b", re.IGNORECASE)
_METRIC = re.compile(
    r"\b(revenue|profit|pbt|pat|ebitda|ebit|dividends?|eps|earnings per share|ratio|margin|"
    r"debt|gearing|assets|liabilities|equity|cash|capex|employees|headcount|rooms|turnover|"
    r"market cap\w*|share price|nav|roe|roce)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d|%|\brs\.?\b|\busd\b|\blkr\b", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\b")  # checked on original casing


class SemanticRouter:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def classify(self, query: str) -> Route:
        if _CONCEPTUAL.search(query):
            return self._log("run_hyde", "explanatory wording", query)

        lookup_signals = sum(bool(p.search(query)) for p in (_LOOKUP_PHRASE, _METRIC, _NUMBER, _ACRONYM))
        if lookup_signals >= 1:
            return self._log("skip_hyde", f"{lookup_signals} fact-lookup signal(s)", query)

        return self._log("run_hyde", "default", query)

    def should_run_hyde(self, query: str) -> bool:
        return self.classify(query) == "run_hyde"

    def _log(self, route: Route, reason: str, query: str) -> Route:
        logger.info(f"[Router] {route} ({reason}): {query[:60]}")
        return route


class RouterTelemetry:
    """Simple counters for routing decisions."""

    def __init__(self):
        self.skipped_hyde_count = 0
        self.ran_hyde_count = 0

    def record(self, route: str):
        if route == "skip_hyde":
            self.skipped_hyde_count += 1
        elif route == "run_hyde":
            self.ran_hyde_count += 1

    def report(self) -> dict:
        total = self.skipped_hyde_count + self.ran_hyde_count
        return {
            "total_queries": total,
            "hyde_skipped": self.skipped_hyde_count,
            "hyde_ran": self.ran_hyde_count,
            "skip_rate_percent": (self.skipped_hyde_count / total * 100) if total else 0.0,
        }
