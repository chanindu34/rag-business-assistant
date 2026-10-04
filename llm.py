"""Quota-aware Gemini text generation.

Free-tier quotas are per model and per day, so a single model can run dry
while others still have headroom. This module:

- tries a chain of models in order (primary first, then fallbacks)
- never retries a DAILY quota error (it cannot recover today); it moves to
  the next model instead
- retries a PER-MINUTE 429 or a 5xx once or twice with jittered backoff,
  honouring the server's suggested delay when it is short
- fails fast on other 4xx errors (bad request, auth), which retries cannot fix
- caches answers on disk so repeated questions cost zero API calls
"""

import hashlib
import json
import logging
import random
import re
import threading
import time
from pathlib import Path
from typing import List, Optional

from google.genai import errors, types

logger = logging.getLogger(__name__)


class QuotaExhaustedError(RuntimeError):
    """Every model in the chain is out of quota or unavailable."""


def _is_daily_quota(err: Exception) -> bool:
    text = str(err)
    return "PerDay" in text or "per day" in text.lower()


def _suggested_delay_seconds(err: Exception) -> Optional[float]:
    match = re.search(r"retry in ([\d.]+)s", str(err))
    return float(match.group(1)) if match else None


class GeminiGateway:
    def __init__(
        self,
        client,
        models: List[str],
        max_transient_retries: int = 2,
        cache_path: Optional[str] = None,
        exhausted_today: Optional[set] = None,
    ):
        self.client = client
        self.models = models
        self.max_transient_retries = max_transient_retries
        # Pass the same set to several gateways so they share what is exhausted.
        self.exhausted_today = exhausted_today if exhausted_today is not None else set()
        self.missing_models = set()
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = self._load_cache()
        self._lock = threading.Lock()

    # ---------- cache ----------
    def _load_cache(self) -> dict:
        if self.cache_path and self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Answer cache unreadable, starting empty.")
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")
        tmp.replace(self.cache_path)  # atomic swap, no half-written file

    @staticmethod
    def _key(prompt: str, purpose: str) -> str:
        return hashlib.sha256(f"{purpose}\n{prompt}".encode("utf-8")).hexdigest()

    # ---------- generation ----------
    def generate(
        self,
        prompt: str,
        purpose: str = "answer",
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cache: bool = True,
    ) -> str:
        key = self._key(prompt, purpose)
        if use_cache and key in self._cache:
            logger.info("[LLM] cache hit (%s)", purpose)
            return self._cache[key]

        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        for model in self.models:
            if model in self.exhausted_today:
                continue
            text = self._try_model(model, prompt, config, purpose)
            if text is not None:
                if use_cache:
                    with self._lock:
                        self._cache[key] = text
                        self._save_cache()
                return text

        missing = [m for m in self.models if m in self.missing_models]
        dry = [m for m in self.models if m not in self.missing_models]
        parts = []
        if dry:
            parts.append("out of quota today: " + ", ".join(dry))
        if missing:
            parts.append("not available on this API key: " + ", ".join(missing))
        raise QuotaExhaustedError("No Gemini model could answer. " + "; ".join(parts))

    def _try_model(self, model, prompt, config, purpose) -> Optional[str]:
        for attempt in range(self.max_transient_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                logger.info("[LLM] %s ok via %s", purpose, model)
                return (response.text or "").strip()
            except errors.ClientError as e:
                if e.code == 429 and _is_daily_quota(e):
                    logger.warning("[LLM] %s daily quota exhausted, trying next model", model)
                    self.exhausted_today.add(model)
                    return None
                if e.code == 429 and attempt < self.max_transient_retries:
                    delay = _suggested_delay_seconds(e) or (2 ** attempt)
                    if delay > 30:  # too long to block a user request
                        logger.warning("[LLM] %s rate limited for %.0fs, trying next model", model, delay)
                        return None
                    time.sleep(delay + random.uniform(0, 1))
                    continue
                if e.code == 404:
                    logger.error("[LLM] model %s not found on this key, skipping it. Run list_models.py", model)
                    self.exhausted_today.add(model)
                    self.missing_models.add(model)
                    return None
                raise  # other 4xx: retrying will not help
            except errors.ServerError:
                if attempt < self.max_transient_retries:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                logger.warning("[LLM] %s server errors, trying next model", model)
                return None
        return None
