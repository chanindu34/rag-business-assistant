"""Single source of truth for settings. Reads config.yaml once.

Any value can be overridden with an environment variable named
RAG_<SECTION>_<KEY>, for example RAG_RETRIEVAL_TOP_K=8.
Secrets (API keys) are NOT here; they stay in .env or platform secrets.
"""

import os
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_ENV_PATH = Path(__file__).parent / ".env"


def _load_dotenv():
    """Load KEY=VALUE pairs from .env into the environment.

    Real environment variables (and platform secrets) always win, so this
    never overrides a key that is already set.
    """
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _load():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for section, values in cfg.items():
        for key, default in values.items():
            if isinstance(default, (list, dict)):
                continue
            override = os.environ.get(f"RAG_{section}_{key}".upper())
            if override is not None:
                values[key] = type(default)(override)
    return cfg


_cfg = _load()

EMBEDDING_MODEL = _cfg["models"]["embedding"]
GENERATION_MODEL = _cfg["models"]["generation"]
GENERATION_FALLBACKS = _cfg["models"].get("generation_fallbacks", [])
HYDE_MODEL = _cfg["models"].get("hyde", GENERATION_MODEL)
HYDE_FALLBACKS = _cfg["models"].get("hyde_fallbacks", [])
JUDGE_MODEL = _cfg["models"]["judge"]
RERANKER_MODEL = _cfg["models"]["reranker"]

COLLECTION_NAME = _cfg["retrieval"]["collection"]
CHROMA_DB_PATH = _cfg["retrieval"]["db_path"]
DEFAULT_TOP_K = _cfg["retrieval"]["top_k"]
USE_HYDE = bool(_cfg["retrieval"].get("use_hyde", True))

PDF_PATH = _cfg["data"]["pdf_path"]
PAGE_RANGES = [tuple(r) for r in _cfg["data"]["page_ranges"]]

CHUNK_SIZE = _cfg["chunking"]["size"]
CHUNK_OVERLAP = _cfg["chunking"]["overlap"]

MAX_RETRIES = _cfg["rate_limits"]["max_retries"]
EMBED_RATE_LIMIT_DELAY_SECONDS = _cfg["rate_limits"]["embed_delay_seconds"]
ANSWER_CACHE_PATH = _cfg["rate_limits"].get("answer_cache_path")

SOURCE_PREVIEW_CHARS = _cfg["ui"]["source_preview_chars"]
SAMPLE_QUESTIONS = _cfg["ui"].get("sample_questions", [])
