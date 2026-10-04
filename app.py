"""
Business Intelligence Assistant

Streamlit chat app that answers questions about John Keells Holdings'
Annual Report 2025/26 using PRODUCTION RAG:
- HyDE (hypothetical document embeddings)
- Hybrid search (BM25 + vector with RRF)
- Cross-encoder reranking
- Confidence guardrails (graceful failure on low confidence)

Includes numbered source citations.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import chromadb
import streamlit as st
from chromadb import EmbeddingFunction
from google import genai

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBED_RATE_LIMIT_DELAY_SECONDS,
    EMBEDDING_MODEL,
    ANSWER_CACHE_PATH,
    GENERATION_FALLBACKS,
    GENERATION_MODEL,
    BM25_WEIGHT,
    HIGH_THRESHOLD,
    HYDE_FALLBACKS,
    HYDE_MODEL,
    LOW_THRESHOLD,
    MAX_QUESTION_CHARS,
    MAX_RETRIES,
    NUM_CANDIDATES,
    USE_HYDE,
    VECTOR_WEIGHT,
    RERANKER_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    RRF_K,
    SAMPLE_QUESTIONS,
    SOURCE_PREVIEW_CHARS,
    parents_path,
)
from retriever import ProductionRAG
from query_condensation import QueryCondenser
from llm import GeminiGateway, QuotaExhaustedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("No API key found. Set GEMINI_API_KEY as an environment variable (local) or in Streamlit Cloud secrets (deployed).")
    st.stop()

from google.genai import types as genai_types

# A hung request must not hang the UI forever. HttpOptions.timeout is in ms.
client_genai = genai.Client(
    api_key=api_key,
    http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
)


def get_embedding(text: str, max_retries: int = MAX_RETRIES) -> List[float]:
    """Embed text via the Gemini embedding API, retrying with exponential backoff."""
    for attempt in range(max_retries):
        try:
            result = client_genai.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            return result.embeddings[0].values
        except Exception:
            logger.warning("Embedding attempt %d/%d failed", attempt + 1, max_retries, exc_info=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to embed after retries.")


def _setting(name: str, default):
    """Environment variable, then Streamlit secret, then config.yaml default.
    Lets a host (e.g. Streamlit Cloud) switch to a smaller reranker without
    a code change."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def validate_question(raw: str):
    """Return (clean_question, error_message). Runs before any API call."""
    q = " ".join((raw or "").split())  # collapse whitespace and newlines
    q = "".join(ch for ch in q if ch.isprintable())
    if not q:
        return None, None
    if len(q) > MAX_QUESTION_CHARS:
        return None, (f"That question is {len(q):,} characters long. Please keep it under "
                      f"{MAX_QUESTION_CHARS} characters and ask about one thing at a time.")
    return q, None


@st.cache_resource
def get_gateways():
    """One gateway for answers, one for HyDE drafts. Shared across sessions."""
    answer_models = [GENERATION_MODEL] + [m for m in GENERATION_FALLBACKS if m != GENERATION_MODEL]
    hyde_models = [HYDE_MODEL] + [m for m in HYDE_FALLBACKS if m != HYDE_MODEL]
    exhausted = set()  # shared: a model that ran dry for HyDE is dry for answers too
    answers = GeminiGateway(
        client_genai, answer_models, cache_path=ANSWER_CACHE_PATH,
        exhausted_today=exhausted, never_cache={"NOT_IN_REPORT"},
    )
    hyde = GeminiGateway(
        client_genai, hyde_models, exhausted_today=exhausted,
        cache_path=ANSWER_CACHE_PATH.replace(".json", "_hyde.json") if ANSWER_CACHE_PATH else None,
    )
    return answers, hyde


def generate_answer(prompt: str) -> str:
    answers, _ = get_gateways()
    return answers.generate(prompt, purpose="answer")


def generate_hyde(prompt: str) -> str:
    _, hyde = get_gateways()
    return hyde.generate(prompt, purpose="hyde", max_output_tokens=200, temperature=0.7)


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by the Gemini embedding API."""

    def __init__(self):
        pass

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            embeddings.append(get_embedding(text))
            time.sleep(EMBED_RATE_LIMIT_DELAY_SECONDS)
        return embeddings


@st.cache_resource
def get_collection():
    """Open the Chroma collection once per process.

    Streamlit reruns this script on every interaction. Creating a new
    PersistentClient on each rerun races on the same SQLite file and causes
    "Could not connect to tenant default_tenant".
    """
    db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    # No embedding function: the app never asks Chroma to embed text. Query
    # vectors are made by the retriever, which also searches in numpy.
    return db_client.get_collection(name=COLLECTION_NAME)


try:
    collection = get_collection()
except Exception:
    logger.exception("Could not open collection %r in %r", COLLECTION_NAME, CHROMA_DB_PATH)
    st.error(
        f"Collection '{COLLECTION_NAME}' was not found in '{CHROMA_DB_PATH}'. "
        "Check config.yaml or re-run ingest.py."
    )
    st.stop()


@st.cache_resource
def load_production_rag():
    """Initialize the production RAG pipeline (cached)."""
    logger.info("Loading ProductionRAG pipeline...")

    all_results = collection.get(include=["documents", "embeddings", "metadatas"])
    chunks = all_results["documents"]
    embeddings = all_results["embeddings"]
    metadatas = all_results["metadatas"]

    parents = {}
    p_path = parents_path(COLLECTION_NAME)
    if os.path.exists(p_path):
        with open(p_path, encoding="utf-8") as f:
            parents = json.load(f)
    else:
        logger.warning("No parent chunks at %s; the answer model will see child chunks only.", p_path)

    # Indexes built by the current ingest.py record the document task type.
    doc_task_type = (collection.metadata or {}).get("doc_task_type")

    if not chunks or embeddings is None or len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Index is empty or inconsistent: {len(chunks or [])} documents, "
            f"{0 if embeddings is None else len(embeddings)} embeddings. Re-run ingest.py."
        )

    logger.info(f"Loaded {len(chunks)} chunks and {len(parents)} parents from {COLLECTION_NAME}")

    rag = ProductionRAG(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=EMBEDDING_MODEL,
        generation_model=GENERATION_MODEL,
        llm_client=client_genai,
        reranker_model=_setting("RAG_MODELS_RERANKER", RERANKER_MODEL),
        hyde_generate_fn=generate_hyde,
        use_hyde=USE_HYDE,
        high_threshold=HIGH_THRESHOLD,
        low_threshold=LOW_THRESHOLD,
        num_candidates=NUM_CANDIDATES,
        rrf_k=RRF_K,
        bm25_weight=BM25_WEIGHT,
        vector_weight=VECTOR_WEIGHT,
        metadatas=metadatas,
        parents=parents,
        doc_task_type=doc_task_type,
        query_task_type="RETRIEVAL_QUERY" if doc_task_type else None,
    )

    return rag


rag_pipeline = load_production_rag()


@st.cache_resource
def get_query_condenser():
    """Initialize query condenser (cached)."""
    return QueryCondenser()


def retrieve(query: str, k: int = DEFAULT_TOP_K, chat_history: List[Dict] = None, on_step=None) -> Dict:
    """Run the retrieval pipeline, resolving follow-up questions first.

    Returns a dict with chunks, retrieval_stats, confidence
    ("high" | "ambiguous" | "low") and previous_question (or None).
    """
    condensed_query, previous_question = get_query_condenser().resolve(chat_history or [], query)
    if previous_question and on_step:
        on_step("Follow-up question, adding context from the previous question")
    result = rag_pipeline.retrieve(condensed_query, top_k=k, on_step=on_step)
    stats = result["retrieval_stats"]
    stats["follow_up"] = previous_question is not None
    sources, seen = [], set()
    for r in result["final_chunks"]:
        if r.get("parent_id") in seen:
            continue  # two matching children in one parent: send the parent once
        seen.add(r.get("parent_id"))
        sources.append({"page": r.get("page"), "text": r.get("context", r["chunk"]), "match": r["chunk"]})
    return {
        "chunks": sources,
        "retrieval_stats": stats,
        "confidence": result["confidence"],
        "previous_question": previous_question,
        "resolved_query": condensed_query,
    }


def _source_text(source) -> str:
    return source["text"] if isinstance(source, dict) else source


def build_prompt(query: str, chunks: List, previous_question: Optional[str] = None) -> str:
    """Build a grounded, citation-instructed prompt from retrieved sources."""
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
If the context does not contain the answer at all, reply with exactly NOT_IN_REPORT and nothing else.
If it answers only part of the question, answer that part and say which part the context doesn't cover.
Cite which source number(s) you used in brackets after each claim, like [1] or [1][3].

<context>
{numbered_context}
</context>

{follow_up}Question: {query}

Answer:"""


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


def answer(query: str, k: int = DEFAULT_TOP_K, on_step=None) -> Dict:
    """Answer a question end to end: resolve follow-up, retrieve, tier, generate."""
    chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
    r = retrieve(query, k, chat_history=chat_history, on_step=on_step)
    # Remember the resolved form on the user message so later follow-ups chain.
    st.session_state.messages[-1]["resolved_query"] = r["resolved_query"]
    chunks, stats, confidence = r["chunks"], r["retrieval_stats"], r["confidence"]
    base = {"retrieval_stats": stats, "confidence": confidence}

    if confidence == "low":
        return {**base, "answer": LOW_CONFIDENCE_MESSAGE, "sources": []}

    prompt = build_prompt(query, chunks, r["previous_question"])
    state = {"declined": False}
    return {**base, "stream": _answer_stream(prompt, confidence, state), "state": state, "sources": chunks}


REFUSAL_TOKEN = "NOT_IN_REPORT"


def _answer_stream(prompt: str, confidence: str, state: Dict):
    """Yield answer text as it is generated.

    The first few characters are held back until we know whether the model
    is refusing (NOT_IN_REPORT). A refusal becomes the standard decline
    message and sets state["declined"], so the UI shows LOW confidence.
    Quota failure becomes an explanatory message.
    """
    answers, _ = get_gateways()
    try:
        pieces = answers.stream(prompt, purpose="answer")
        head = ""
        for piece in pieces:
            head += piece
            if len(head.strip()) >= len(REFUSAL_TOKEN) or not REFUSAL_TOKEN.startswith(head.strip()[:len(REFUSAL_TOKEN)]):
                break
        if head.strip().startswith(REFUSAL_TOKEN):
            state["declined"] = True
            yield LOW_CONFIDENCE_MESSAGE
            return
        yield head
        yield from pieces
    except QuotaExhaustedError as e:
        logger.warning("%s", e)
        yield (f"I couldn't write an answer right now. {e}. "
               "The sources I found are listed below. Quotas reset daily "
               "(https://ai.dev/rate-limit); run `python3 list_models.py` to see "
               "which models your key can use.")
        return
    if confidence == "ambiguous":
        yield AMBIGUOUS_NOTE


TIER_BADGE = {
    "high": ("green", "High confidence"),
    "ambiguous": ("orange", "Moderate confidence"),
    "low": ("red", "Not found in report"),
}


def render_answer_meta(stats: Dict, confidence: str) -> None:
    """Badges under each answer: confidence, retrieval path, timing."""
    color, label = TIER_BADGE.get(confidence, ("gray", confidence.title()))
    if "top_score" in stats and not stats.get("declined_by_model"):
        label += f" · {stats['top_score']:.2f}"
    badges = [f":{color}-badge[{label}]"]
    if stats.get("dense_unavailable"):
        badges.append(":orange-badge[Keyword search only]")
    elif stats.get("hyde_used"):
        badges.append(":blue-badge[HyDE]")
    elif stats.get("route") == "skip_hyde":
        badges.append(":blue-badge[Fact lookup, HyDE skipped]")
    if stats.get("follow_up"):
        badges.append(":violet-badge[Follow-up resolved]")
    if stats.get("total_ms"):
        badges.append(f":gray-badge[{stats['total_ms'] / 1000:.1f}s]")
    st.markdown(" ".join(badges))


def render_sources(sources: List) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, source in enumerate(sources, 1):
            page = source.get("page") if isinstance(source, dict) else None
            tag = f":gray-badge[Page {page}]" if page else ""
            st.markdown(f"**[{i}]** {tag}  \n{format_source(source, with_page=False)}")


def format_source(source, max_chars: int = SOURCE_PREVIEW_CHARS, with_page: bool = True) -> str:
    """One source line: page number plus the passage that matched the question."""
    if isinstance(source, dict):
        text = source.get("match") or source.get("text", "")
        page = f"*Page {source['page']}* · " if with_page and source.get("page") else ""
    else:  # messages saved before page metadata existed
        text, page = source, ""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return page + text


def _queue_question(question: str) -> None:
    """Button callback: queue a sample question to be answered on this rerun."""
    st.session_state.pending_query = question


def _new_chat() -> None:
    st.session_state.messages = []


def render_assistant_message(message: Dict) -> None:
    st.markdown(message["content"])
    if message.get("retrieval_stats"):
        render_answer_meta(message["retrieval_stats"], message.get("confidence", "high"))
    render_sources(message.get("sources"))


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="BI Assistant", layout="centered", initial_sidebar_state="collapsed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0;'>Business Intelligence Assistant</h1>"
        "<p style='text-align: center; color: gray; font-size: 1.1rem; margin-top: 0;'>"
        "for John Keells Holdings</p>",
        unsafe_allow_html=True,  # static text only, no user input
    )
    if st.session_state.messages:
        _, action = st.columns([5, 1])
        action.button("New chat", on_click=_new_chat, use_container_width=True)

    # Empty state: sample questions until the first question is asked.
    if SAMPLE_QUESTIONS and not st.session_state.messages and "pending_query" not in st.session_state:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, question in enumerate(SAMPLE_QUESTIONS):
            cols[i % 2].button(
                question,
                key=f"sample_main_{i}",
                on_click=_queue_question,
                args=(question,),
                use_container_width=True,
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    typed_query = st.chat_input("Ask about results, strategy, risks, outlook...", max_chars=MAX_QUESTION_CHARS * 2)
    query, input_error = validate_question(typed_query or st.session_state.pop("pending_query", None))
    if input_error:
        st.warning(input_error)
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        started = time.time()
        # Live pipeline progress: each retrieval stage reports as it starts.
        # Steps are visible while it works, then fold away once the answer starts.
        with st.status("Searching the report...", expanded=True) as status:
            result = answer(query, on_step=status.write)
            stats = result["retrieval_stats"]
            n = len(result["sources"])
            status.update(
                label=(f"Retrieved {n} passage{'s' if n != 1 else ''} in {time.time() - started:.1f}s"
                       if n else f"No matching passages ({time.time() - started:.1f}s)"),
                state="complete",
                expanded=False,
            )
        sources, confidence = result["sources"], result.get("confidence", "high")

        if "stream" in result:
            # Text appears as the model writes it instead of all at once.
            answer_text = st.write_stream(result["stream"])
            if result["state"]["declined"]:
                confidence, sources = "low", []
                stats["declined_by_model"] = True
        else:
            answer_text = result["answer"]
            st.markdown(answer_text)
        stats["total_ms"] = (time.time() - started) * 1000

        message = {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "retrieval_stats": stats,
            "confidence": confidence,
        }
        render_answer_meta(stats, confidence)
        render_sources(sources)
    st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
