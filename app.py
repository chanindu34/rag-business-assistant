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

import logging
import os
import time
from typing import Dict, List

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
    HYDE_FALLBACKS,
    HYDE_MODEL,
    MAX_RETRIES,
    USE_HYDE,
    RERANKER_MODEL,
    SAMPLE_QUESTIONS,
    SOURCE_PREVIEW_CHARS,
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

client_genai = genai.Client(api_key=api_key)


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


@st.cache_resource
def get_gateways():
    """One gateway for answers, one for HyDE drafts. Shared across sessions."""
    answer_models = [GENERATION_MODEL] + [m for m in GENERATION_FALLBACKS if m != GENERATION_MODEL]
    hyde_models = [HYDE_MODEL] + [m for m in HYDE_FALLBACKS if m != HYDE_MODEL]
    exhausted = set()  # shared: a model that ran dry for HyDE is dry for answers too
    answers = GeminiGateway(client_genai, answer_models, cache_path=ANSWER_CACHE_PATH, exhausted_today=exhausted)
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
    return db_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction()
    )


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

    all_results = collection.get(include=["documents", "embeddings"])
    chunks = all_results["documents"]
    embeddings = all_results["embeddings"]

    if not chunks or embeddings is None or len(chunks) != len(embeddings):
        raise RuntimeError(
            f"Index is empty or inconsistent: {len(chunks or [])} documents, "
            f"{0 if embeddings is None else len(embeddings)} embeddings. Re-run ingest.py."
        )

    logger.info(f"Loaded {len(chunks)} chunks for production RAG")

    rag = ProductionRAG(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=EMBEDDING_MODEL,
        generation_model=GENERATION_MODEL,
        llm_client=client_genai,
        reranker_model=RERANKER_MODEL,
        hyde_generate_fn=generate_hyde,
        use_hyde=USE_HYDE,
    )

    return rag


rag_pipeline = load_production_rag()


@st.cache_resource
def get_query_condenser():
    """Initialize query condenser (cached)."""
    return QueryCondenser()


def retrieve(query: str, k: int = DEFAULT_TOP_K, chat_history: List[Dict] = None) -> tuple[List[str], Dict, str]:
    """Retrieve using production RAG: HyDE + Hybrid + Rerank.

    Supports multi-turn chat by condensing history with current query.

    Returns:
        (chunks, retrieval_stats, confidence_level)
        confidence_level: "high" or "low"
    """
    # Condense query with chat history for multi-turn support
    condenser = get_query_condenser()
    if chat_history:
        condensed_query = condenser.condense(chat_history, query)
        logger.info(f"[Multi-turn] Condensed '{query[:50]}...' with history")
    else:
        condensed_query = query

    # Use condensed query for retrieval
    result = rag_pipeline.retrieve(condensed_query, top_k=k, verbose=False)
    chunks = [r["chunk"] for r in result["final_chunks"]]
    confidence = result.get("confidence", "high")
    return chunks, result["retrieval_stats"], confidence


def build_prompt(query: str, chunks: List[str]) -> str:
    """Build a grounded, citation-instructed prompt from retrieved chunks."""
    numbered_context = ""
    for i, chunk in enumerate(chunks):
        numbered_context += f"[{i+1}] {chunk}\n\n"

    prompt = f"""Answer the question using ONLY the information in the context below.
Do NOT use any outside knowledge, even if you recognize the company or topic.
If specific details aren't in the context, explicitly say "the provided context doesn't cover this" rather than filling gaps from general knowledge.
Cite which source number(s) you used in brackets after each claim, like [1] or [1][3].

<context>
{numbered_context}
</context>

Question: {query}

Answer:"""
    return prompt


def answer(query: str, k: int = DEFAULT_TOP_K) -> Dict:
    """Answer a question end to end: retrieve -> prompt -> generate.

    If retrieval confidence is low, returns a graceful fallback message.
    Handles multi-turn chat by using chat history for context.
    """
    # Get chat history (excluding current query)
    chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []

    chunks, retrieval_stats, confidence = retrieve(query, k, chat_history=chat_history)

    if confidence == "low":
        return {
            "answer": "I do not have sufficient internal documentation to answer this question with confidence. Please try:\n1. Rephrasing your question more specifically\n2. Checking if the information exists in the annual report\n3. Searching online for additional context",
            "sources": [],
            "retrieval_stats": retrieval_stats,
            "confidence": "low",
        }

    prompt = build_prompt(query, chunks)
    try:
        answer_text = generate_answer(prompt)
    except QuotaExhaustedError as e:
        logger.warning("%s", e)
        return {
            "answer": f"I couldn't write an answer right now. {e}. "
                      "The sources I found are listed below. "
                      "Quotas reset daily (https://ai.dev/rate-limit); "
                      "run `python list_models.py` to see which models your key can use.",
            "sources": chunks,
            "retrieval_stats": retrieval_stats,
            "confidence": confidence,
        }
    return {
        "answer": answer_text,
        "sources": chunks,
        "retrieval_stats": retrieval_stats,
        "confidence": "high",
    }


def format_source(chunk: str, max_chars: int = SOURCE_PREVIEW_CHARS) -> str:
    """Format a source chunk for display."""
    if len(chunk) > max_chars:
        return chunk[:max_chars] + "..."
    return chunk


def _queue_question(question: str) -> None:
    """Button callback: queue a sample question to be answered on this rerun."""
    st.session_state.pending_query = question


def render_sample_questions(location) -> None:
    """Render sample question buttons in the given container."""
    for i, question in enumerate(SAMPLE_QUESTIONS):
        location.button(
            question,
            key=f"sample_{location is st.sidebar}_{i}",
            on_click=_queue_question,
            args=(question,),
            use_container_width=True,
        )


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="BI Assistant", layout="wide")
    st.title("Business Intelligence Assistant")
    st.markdown("Ask questions about John Keells Holdings' 2025/26 Annual Report")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if SAMPLE_QUESTIONS:
        st.sidebar.subheader("Try asking")
        render_sample_questions(st.sidebar)

        # Empty state: show the samples in the main area until the first question.
        if not st.session_state.messages and "pending_query" not in st.session_state:
            st.markdown("**Not sure where to start? Try one of these:**")
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
            st.markdown(message["content"])
            if message.get("retrieval_stats"):
                stats = message["retrieval_stats"]
                confidence = message.get("confidence", "high")
                confidence_color = "🟢" if confidence == "high" else "🔴"
                st.caption(
                    f"{confidence_color} Confidence: {confidence.upper()} | "
                    f"Retrieval: {stats['method']} | "
                    f"Candidates: {stats['num_candidates_evaluated']} | "
                    f"Rerank: {stats['rerank_latency_ms']:.0f}ms"
                )
            if message.get("sources"):
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[{i}]** {format_source(source)}")

    typed_query = st.chat_input("Ask a question...")
    query = typed_query or st.session_state.pop("pending_query", None)
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer(query)
                answer_text = result["answer"]
                sources = result["sources"]
                retrieval_stats = result["retrieval_stats"]
                confidence = result.get("confidence", "high")

            st.markdown(answer_text)

            confidence_color = "🟢" if confidence == "high" else "🔴"
            st.caption(
                f"{confidence_color} Confidence: {confidence.upper()} | "
                f"Retrieval: {retrieval_stats['method']} | "
                f"Candidates: {retrieval_stats['num_candidates_evaluated']} | "
                f"Rerank: {retrieval_stats['rerank_latency_ms']:.0f}ms"
            )

            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**[{i}]** {format_source(source)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "retrieval_stats": retrieval_stats,
                "confidence": confidence,
            })


if __name__ == "__main__":
    main()
