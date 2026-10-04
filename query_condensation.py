"""
Query condensation for multi-turn chat, with zero API calls and no model.

A follow-up like "Why did it grow so much?" is useless for retrieval on its
own. If the new question looks like a follow-up, the previous user question
is prepended so retrieval sees "How much did EBITDA grow? Why did it grow so
much?".

Follow-up detection is a heuristic, deliberately conservative:
- Strong references anywhere: it, its, they, them, their, he, she, former,
  latter, same.
- Weak references (this, that, these, those) only when they stand alone:
  at the end ("why is that?") or before a generic noun ("that segment").
  "the risks that the company faces" is NOT a follow-up.
- Time phrases ("this year", "last year") are ignored.
- Openers like "and", "also", "what about", "how about", "compared to".

Upgrade path: an LLM rewrite handles more cases but costs an API call per
follow-up.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TIME_PHRASE = re.compile(
    r"\b(this|that|last|next|previous|the same)\s+(financial\s+|fiscal\s+)?(year|quarter|period|month)\b",
    re.IGNORECASE,
)
_STRONG_REF = re.compile(r"\b(it|its|they|them|their|he|she|his|her|former|latter|same)\b", re.IGNORECASE)
_GENERIC_NOUNS = (
    r"one|ones|figure|figures|number|numbers|amount|segment|sector|business|company|"
    r"investment|project|increase|decrease|growth|change|result|results|trend|ratio|metric"
)
_WEAK_REF = re.compile(
    rf"\b(this|that|these|those)\b(?=\s*(?:[?.!,]|$|\s+(?:{_GENERIC_NOUNS})\b))",
    re.IGNORECASE,
)
_CONTINUATION = re.compile(r"^\s*(and|also|what about|how about|compared (?:to|with)|vs\.?|versus)\b", re.IGNORECASE)


class QueryCondenser:
    def is_follow_up(self, query: str) -> bool:
        text = _TIME_PHRASE.sub(" ", query)
        return bool(_CONTINUATION.search(text) or _STRONG_REF.search(text) or _WEAK_REF.search(text))

    @staticmethod
    def last_user_question(chat_history: List[Dict]) -> Optional[str]:
        """Most recent user question, in its resolved form when available.

        Using the resolved form lets follow-ups chain: after
        "How much did EBITDA grow?" -> "Why did it grow?", a third
        "And in Retail?" still carries the EBITDA context.
        """
        for msg in reversed(chat_history or []):
            if msg.get("role") == "user" and msg.get("content"):
                return msg.get("resolved_query") or msg["content"]
        return None

    def resolve(self, chat_history: List[Dict], current_query: str) -> Tuple[str, Optional[str]]:
        """Return (query_for_retrieval, previous_question_or_None)."""
        previous = self.last_user_question(chat_history)
        if previous and self.is_follow_up(current_query):
            previous = " ".join(previous.split()[-60:])  # cap chained context
            condensed = f"{previous} {current_query}"
            logger.info(f"[Condense] follow-up detected, using: {condensed[:120]}")
            return condensed, previous
        return current_query, None

    def condense(self, chat_history: List[Dict], current_query: str) -> str:
        return self.resolve(chat_history, current_query)[0]
