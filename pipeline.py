"""
Everything needed to answer a question, without any UI.

app.py (Streamlit), evaluate.py (offline evaluation) and the unit tests all
build the pipeline from here, so they exercise exactly the same code.
"""

import json
import logging
import os
import re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from config import (
    ANSWER_CACHE_PATH,
    BM25_WEIGHT,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GENERATION_FALLBACKS,
    GENERATION_MODEL,
    HIGH_THRESHOLD,
    HYDE_FALLBACKS,
    HYDE_MODEL,
    LOW_THRESHOLD,
    MAX_QUESTION_CHARS,
    NUM_CANDIDATES,
    REQUEST_TIMEOUT_SECONDS,
    RERANKER_MODEL,
    RRF_K,
    USE_HYDE,
    VECTOR_WEIGHT,
    parents_path,
)

logger = logging.getLogger(__name__)

REFUSAL_TOKEN = "NOT_IN_REPORT"
LOW_CONFIDENCE_MESSAGE = (
    "I do not have sufficient internal documentation to answer this question with confidence. Please try:\n"
    "1. Rephrasing your question more specifically\n"
    "2. Checking if the information exists in the annual report\n"
    "3. Searching online for additional context"
)
AMBIGUOUS_NOTE = (
    "\n\n> **Moderate confidence.** The retrieved passages only partly match this question. "
    "Check the sources below before relying on this answer."
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def make_client(api_key: Optional[str] = None):
    """Gemini client with a per-request timeout (HttpOptions.timeout is in ms)."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key or os.environ.get("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )


def make_gateways(client, use_cache: bool = True):
    """(answers, hyde) gateways. They share the set of exhausted models, and
    HyDE never falls back to the answer models, so it cannot eat their quota."""
    from llm import GeminiGateway

    answer_models = [GENERATION_MODEL] + [m for m in GENERATION_FALLBACKS if m != GENERATION_MODEL]
    hyde_models = [HYDE_MODEL] + [m for m in HYDE_FALLBACKS if m != HYDE_MODEL]
    exhausted = set()
    answers = GeminiGateway(
        client, answer_models,
        cache_path=ANSWER_CACHE_PATH if use_cache else None,
        exhausted_today=exhausted, never_cache={REFUSAL_TOKEN},
    )
    hyde = GeminiGateway(
        client, hyde_models, exhausted_today=exhausted,
        cache_path=(ANSWER_CACHE_PATH.replace(".json", "_hyde.json") if ANSWER_CACHE_PATH and use_cache else None),
    )
    return answers, hyde


def load_index(collection_name: str = COLLECTION_NAME) -> Dict:
    """Chunks, vectors, metadata and parents for one collection."""
    import chromadb

    collection = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(name=collection_name)
    r = collection.get(include=["documents", "embeddings", "metadatas"])
    chunks, embeddings = r["documents"], r["embeddings"]
    if not chunks or embeddings is None or len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Index is empty or inconsistent: {len(chunks or [])} documents, "
            f"{0 if embeddings is None else len(embeddings)} embeddings. Re-run ingest.py."
        )
    parents = {}
    p_path = parents_path(collection_name)
    if os.path.exists(p_path):
        with open(p_path, encoding="utf-8") as f:
            parents = json.load(f)
    else:
        logger.warning("No parent chunks at %s; the answer model will see child chunks only.", p_path)
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "metadatas": r["metadatas"],
        "parents": parents,
        # Indexes built by the current ingest.py record the document task type.
        "doc_task_type": (collection.metadata or {}).get("doc_task_type"),
    }


def build_rag(client, index: Dict, hyde_generate_fn=None, reranker_model: str = RERANKER_MODEL):
    from retriever import ProductionRAG

    logger.info(f"Loaded {len(index['chunks'])} chunks and {len(index['parents'])} parents from {COLLECTION_NAME}")
    return ProductionRAG(
        chunks=index["chunks"],
        embeddings=index["embeddings"],
        embedding_model=EMBEDDING_MODEL,
        generation_model=GENERATION_MODEL,
        llm_client=client,
        reranker_model=reranker_model,
        hyde_generate_fn=hyde_generate_fn,
        use_hyde=USE_HYDE,
        high_threshold=HIGH_THRESHOLD,
        low_threshold=LOW_THRESHOLD,
        num_candidates=NUM_CANDIDATES,
        rrf_k=RRF_K,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
        metadatas=index["metadatas"],
        parents=index["parents"],
        doc_task_type=index["doc_task_type"],
        query_task_type="RETRIEVAL_QUERY" if index["doc_task_type"] else None,
    )


# ---------------------------------------------------------------------------
# Per-question helpers (pure functions, unit tested)
# ---------------------------------------------------------------------------
def validate_question(raw: Optional[str], max_chars: int = MAX_QUESTION_CHARS) -> Tuple[Optional[str], Optional[str]]:
    """Return (clean_question, error_message). Runs before any API call."""
    q = " ".join((raw or "").split())  # collapse whitespace and newlines
    q = "".join(ch for ch in q if ch.isprintable())
    if not q:
        return None, None
    if len(q) > max_chars:
        return None, (f"That question is {len(q):,} characters long. Please keep it under "
                      f"{max_chars} characters and ask about one thing at a time.")
    return q, None


def sources_from_result(final_chunks: List[Dict]) -> List[Dict]:
    """Reranked children -> parent passages for the prompt, deduplicated, in rank order."""
    sources, seen = [], set()
    for r in final_chunks:
        if r.get("parent_id") in seen:
            continue  # two matching children in one parent: send the parent once
        seen.add(r.get("parent_id"))
        sources.append({"page": r.get("page"), "text": r.get("context", r["chunk"]), "match": r["chunk"]})
    return sources


def _source_text(source) -> str:
    return source["text"] if isinstance(source, dict) else source


def build_prompt(query: str, chunks: List, previous_question: Optional[str] = None) -> str:
    """Grounded, citation-instructed prompt with an explicit refusal contract."""
    lines = []
    for i, c in enumerate(chunks, 1):
        page = c.get("page") if isinstance(c, dict) else None
        label = f"[{i}] (page {page})" if page else f"[{i}]"
        lines.append(f"{label} {_source_text(c)}\n\n")
    numbered_context = "".join(lines)
    follow_up = (
        f'This is a follow-up to the earlier question: "{previous_question}". '
        "Interpret references like \"it\" or \"that\" accordingly.\n\n"
        if previous_question else ""
    )
    return f"""Answer the question using ONLY the information in the context below.
Do NOT use any outside knowledge, even if you recognize the company or topic.
If the context does not contain the answer at all, reply with exactly {REFUSAL_TOKEN} and nothing else.
If it answers only part of the question, answer that part and say which part the context doesn't cover.
Cite which source number(s) you used in brackets after each claim, like [1] or [1][3].

<context>
{numbered_context}
</context>

{follow_up}Question: {query}

Answer:"""


def peek_refusal(pieces: Iterable[str], token: str = REFUSAL_TOKEN) -> Tuple[bool, Iterator[str]]:
    """Look at the start of a streamed answer to see if it is the refusal token.

    Holds back only as many characters as needed to decide, so a normal
    answer still streams immediately. Returns (refused, remaining_pieces);
    when refused, remaining_pieces is empty.
    """
    it = iter(pieces)
    head = ""
    for piece in it:
        head += piece
        stripped = head.strip()
        if len(stripped) >= len(token) or not token.startswith(stripped):
            break
    if head.strip().startswith(token):
        return True, iter(())

    def rest():
        if head:
            yield head
        yield from it

    return False, rest()


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-.!|~$])")


def escape_markdown(text: str) -> str:
    """Show user text literally: '9**9**9' must not render as a bold '999'."""
    return _MD_SPECIAL.sub(r"\\\1", text)


# '*' or '**' with a number or ')' just before (spaces allowed) and a number or '(' after.
_ARITH_STARS = re.compile(r"(?<=[\d)])(\s*)(\*{1,2})(?=\s*[\d(])")


def plain_answer(text: str) -> str:
    """Keep model maths literal while leaving its bold and lists alone.

    '$...$' would render as LaTeX, and '*' or '**' between numbers ('9**9**9',
    '2 * 3') would turn into bold or italics. Safe to apply twice.
    """
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    return _ARITH_STARS.sub(lambda m: m.group(1) + "\\*" * len(m.group(2)), text)

