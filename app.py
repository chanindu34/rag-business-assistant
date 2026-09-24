"""
Business Intelligence Assistant

Streamlit chat app that answers questions about John Keells Holdings'
Annual Report 2025/26 using retrieval-augmented generation over a
ChromaDB vector store, with numbered source citations.
"""

import logging
import os
import time
from typing import Dict, List

import chromadb
import streamlit as st
from chromadb import EmbeddingFunction
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
COLLECTION_NAME = "day10_documents"
CHROMA_DB_PATH = "chroma_db"

DEFAULT_TOP_K = 6
MAX_RETRIES = 5
EMBED_RATE_LIMIT_DELAY_SECONDS = 0.7
SOURCE_PREVIEW_CHARS = 300

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


def generate_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> str:
    """Generate a response via the Gemini generation API, retrying with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client_genai.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            return response.text
        except Exception:
            logger.warning("Generation attempt %d/%d failed", attempt + 1, max_retries, exc_info=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to generate after retries.")


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function backed by the Gemini embedding API."""

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            embeddings.append(get_embedding(text))
            time.sleep(EMBED_RATE_LIMIT_DELAY_SECONDS)
        return embeddings


db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

collection = db_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=GeminiEmbeddingFunction()
)


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[str]:
    """Retrieve the top-k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


def build_prompt(query: str, chunks: List[str]) -> str:
    """Build a grounded, citation-instructed prompt from retrieved chunks."""
    numbered_context = ""
    for i, chunk in enumerate(chunks):
        numbered_context += f"[{i+1}] {chunk}\n\n"

    prompt = f"""Answer the question using ONLY the information in the context below.
Do NOT use any outside knowledge, even if you recognize the company or topic.
If specific details aren't in the context, explicitly say "the provided context doesn't cover this" rather than filling gaps from general knowledge.
Cite which source number(s) you used in brackets after each claim, like [1] or [1][3].

Context:
{numbered_context}

Question: {query}

Answer:"""
    return prompt


def answer(query: str, k: int = DEFAULT_TOP_K) -> Dict:
    """Answer a question end to end: retrieve, build prompt, generate."""
    chunks = retrieve(query, k)
    prompt = build_prompt(query, chunks)
    answer_text = generate_with_retry(prompt)
    return {"answer": answer_text, "sources": chunks}


# --- Streamlit UI ---

st.markdown("""
<style>
:root {
    --bg: #0E1117;
    --bg-secondary: #1A1D24;
    --text: #E5E7EB;
    --accent: #3B82F6;
    --accent-soft: rgba(59, 130, 246, 0.12);
    --border: rgba(255, 255, 255, 0.08);
}

.stApp {
    background-color: var(--bg);
    color: var(--text);
}

[data-testid="stChatMessage"] {
    background-color: var(--bg-secondary);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 0.4rem 0.2rem;
    margin-bottom: 0.6rem;
}

.stButton button {
    background-color: var(--bg-secondary);
    color: var(--text);
    border-radius: 8px;
    border: 1px solid var(--border);
    text-align: left;
    transition: all 0.15s ease;
}

.stButton button:hover {
    border-color: var(--accent);
    color: var(--accent);
    background-color: var(--accent-soft);
}

[data-testid="stExpander"] {
    background-color: var(--bg-secondary);
    border-radius: 8px;
    border: 1px solid var(--border);
}

[data-testid="stExpander"] summary {
    color: var(--accent);
    font-weight: 500;
}

[data-testid="stChatInput"] {
    background-color: var(--bg-secondary);
}

.citation-badge {
    font-size: 0.72rem;
    font-weight: 600;
    background-color: var(--accent-soft);
    color: var(--accent);
    padding: 1px 7px;
    border-radius: 4px;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("Business Intelligence Assistant")
st.caption("Ask questions about John Keells Holdings' Annual Report 2025/26")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.write("Try an example question:")
    example_questions = [
        "What was the Group's EBITDA growth this year?",
        "What are the biggest risks facing the company?",
        "What is the outlook for Sri Lanka's tourism sector?",
        "What is the Group's net debt to EBITDA ratio?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(q, use_container_width=True, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    try:
                        result = answer(q)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })
                    except Exception:
                        logger.error("Failed to answer example question: %r", q, exc_info=True)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Sorry, I hit a temporary error reaching the AI service. Please try asking again.",
                            "sources": []
                        })
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(
                        f'<span class="citation-badge">{i+1}</span> {source[:SOURCE_PREVIEW_CHARS]}...',
                        unsafe_allow_html=True
                    )

user_question = st.chat_input("Ask a question about the report...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer(user_question)
                st.write(result["answer"])

                with st.expander("Sources"):
                    for i, source in enumerate(result["sources"]):
                        st.markdown(
                            f'<span class="citation-badge">{i+1}</span> {source[:SOURCE_PREVIEW_CHARS]}...',
                            unsafe_allow_html=True
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception:
                logger.error("Failed to answer user question: %r", user_question, exc_info=True)
                error_msg = "Sorry, I hit a temporary error reaching the AI service. Please try asking again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
