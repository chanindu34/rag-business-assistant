"""
Business Intelligence Assistant: Streamlit UI.

Answers questions about John Keells Holdings' Annual Report 2025/26 with
page citations. The pipeline itself (routing, HyDE, hybrid search, reranking,
confidence tiers, prompt and refusal handling) lives in pipeline.py and
retriever.py; this file is only the chat interface around it.
"""

import hashlib
import logging
import os
import time
from typing import Dict, List

import streamlit as st

from config import DEFAULT_TOP_K, MAX_QUESTION_CHARS, RERANKER_MODEL, SAMPLE_QUESTIONS, SOURCE_PREVIEW_CHARS
from llm import QuotaExhaustedError
from pipeline import (
    AMBIGUOUS_NOTE,
    LOW_CONFIDENCE_MESSAGE,
    build_prompt,
    build_rag,
    load_index,
    make_client,
    make_gateways,
    escape_markdown,
    peek_refusal,
    plain_answer,
    sources_from_result,
    validate_question,
)
from query_condensation import QueryCondenser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _setting(name: str, default):
    """Environment variable, then Streamlit secret, then config.yaml default.
    Lets a host (e.g. Streamlit Cloud) switch to a smaller reranker or set the
    API key without a code change."""
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


api_key = _setting("GEMINI_API_KEY", None)
if not api_key:
    st.error("No API key found. Set GEMINI_API_KEY as an environment variable (local) or in Streamlit Cloud secrets (deployed).")
    st.stop()

client_genai = make_client(api_key)


def _code_version() -> str:
    """Fingerprint of the files that shape the cached objects.

    st.cache_resource keys on the cached function's own source, so after a
    deploy that only changes retriever.py the old pipeline object (with the
    old methods) would be reused. Passing this fingerprint as an argument
    forces a rebuild whenever any of these files change.
    """
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("pipeline.py", "retriever.py", "semantic_router.py", "confidence_tiers.py", "llm.py", "config.yaml"):
        try:
            with open(os.path.join(here, name), "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


@st.cache_resource
def _get_gateways(code_version: str):
    """Answer and HyDE gateways, shared across sessions, rebuilt on code change."""
    return make_gateways(client_genai)


def get_gateways():
    return _get_gateways(_code_version())


def generate_hyde(prompt: str) -> str:
    _, hyde = get_gateways()
    return hyde.generate(prompt, purpose="hyde", max_output_tokens=200, temperature=0.7)


@st.cache_resource
def load_production_rag(code_version: str):
    """Build the pipeline once per process and code version.

    Opening Chroma once here also avoids the "Could not connect to tenant
    default_tenant" race that happens when every Streamlit rerun opens it.
    """
    logger.info("Loading ProductionRAG pipeline...")
    return build_rag(
        client_genai,
        load_index(),
        hyde_generate_fn=generate_hyde,
        reranker_model=_setting("RAG_MODELS_RERANKER", RERANKER_MODEL),
    )


try:
    rag_pipeline = load_production_rag(_code_version())
except Exception:
    logger.exception("Could not load the index")
    st.error("The search index could not be loaded. Check `retrieval.collection` in config.yaml or re-run ingest.py.")
    st.stop()


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
    sources = sources_from_result(result["final_chunks"])
    return {
        "chunks": sources,
        "retrieval_stats": stats,
        "confidence": result["confidence"],
        "previous_question": previous_question,
        "resolved_query": condensed_query,
    }


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


def _answer_stream(prompt: str, confidence: str, state: Dict):
    """Yield answer text as it is generated.

    A NOT_IN_REPORT refusal becomes the standard decline message and sets
    state["declined"], so the UI shows it as not found. Quota failure becomes
    an explanatory message.
    """
    answers, _ = get_gateways()
    try:
        refused, pieces = peek_refusal(answers.stream(prompt, purpose="answer"))
        if refused:
            state["declined"] = True
            yield LOW_CONFIDENCE_MESSAGE
            return
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
    st.markdown(plain_answer(message["content"]))
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
                st.markdown(escape_markdown(message["content"]))

    typed_query = st.chat_input("Ask about results, strategy, risks, outlook...", max_chars=MAX_QUESTION_CHARS * 2)
    query, input_error = validate_question(typed_query or st.session_state.pop("pending_query", None))
    if input_error:
        st.warning(input_error)
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(escape_markdown(query))

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
            answer_text = st.write_stream(plain_answer(piece) for piece in result["stream"])
            if result["state"]["declined"]:
                confidence, sources = "low", []
                stats["declined_by_model"] = True
        else:
            answer_text = result["answer"]
            st.markdown(plain_answer(answer_text))
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
