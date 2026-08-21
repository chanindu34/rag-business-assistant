# Business Intelligence Assistant

A retrieval-augmented generation (RAG) system that answers natural-language questions about John Keells Holdings' Annual Report 2025/26, with source citations for every claim.

**Live demo:** https://rag-business-assistant-pdpsjsuaxzvmaaff8edv3b.streamlit.app

## What it does

A chat interface over a real 612-page annual report. Ask a question, get an answer synthesized from the actual document — not a general-knowledge guess — with numbered citations you can click to verify against the exact source text.

## Architecture

```
PDF → chunk (500 chars, 50 overlap) → embed (Gemini) → store (ChromaDB)
                                                              ↓
query → embed → retrieve top-k → build cited prompt → generate answer
```

- **Embeddings:** Gemini `gemini-embedding-001`, via a custom adapter class decoupling the vector store from any one provider
- **Vector store:** ChromaDB, persistent, with resumable batch ingestion (survives API failures mid-run without losing progress)
- **Generation:** Gemini `gemini-2.5-flash`, with exponential-backoff retry logic on both embedding and generation calls
- **Chunking:** LangChain's RecursiveCharacterTextSplitter — chunk size chosen empirically (tested at 200/500/1000 chars; 500 was the only size that avoided both mid-sentence cuts and loss of retrieval precision)
- **Document scope:** deliberately limited to pages 0-60 and 138-160 (core financial/strategic narrative + outlook). Two reasons: pages 166-293 repeat Group-level facts at finer industry-group granularity, and the regulatory disclosure sections (294+) are structurally tabular (Topic | Metric Code | Unit of Measure) — standard text extraction flattens tables into disconnected values, losing the relational structure that gives a table its meaning. Scoping to narrative-heavy sections also kept ingestion within the embedding API's daily free-tier quota.

## Testing results

10 real questions, manually graded against the actual source text (not assumed correct):

- **9/10 answered correctly**, including hard synthesis questions with zero keyword overlap with the source phrasing
- **3/10 correct but repetitive** — multiple retrieved chunks restating the same point in slightly different words
- **1/10 contained a confirmed hallucination** — cited a specific initiative name that was verifiably absent from all retrieved context

### The hallucination, and what I learned from it

Asked about ESG initiatives, the model referenced a specific program by name that wasn't present in any of the 6 retrieved chunks — confirmed via direct string search, not assumption. I tried two fixes:

1. **Explicit prompt instruction** forbidding outside knowledge, even when the model recognizes the company — did not resolve it on retest.
2. **A regex-based faithfulness check** flagging capitalized terms absent from retrieved context — produced false positives on ordinary capitalized words while still missing the actual hallucinated term.

**Conclusion:** this is a known, real limitation of prompt-only grounding — LLMs can surface training-data knowledge about well-known public entities regardless of instructions. A production fix would need semantic-level faithfulness verification (comparing claim embeddings against source embeddings), not prompt engineering or string matching. I removed the unreliable regex check from the live app rather than ship something that produces misleading signals, documenting the finding here instead.

### Known limitation: no multi-turn memory

Each question is answered independently — retrieval and generation don't incorporate prior conversation turns. Follow-up questions using pronouns or references ("that one", "the other option") can't be resolved, and the system correctly says so rather than guessing. A production version would need to include recent conversation history in the retrieval/generation context.

## Tech stack

Python · Google Gemini API · ChromaDB · LangChain (text splitting) · Streamlit

## Run it locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

## What I'd build next

- **Layout-aware document parsing** (e.g. LlamaParse) to properly handle the tabular sections that are currently excluded — preserving table structure as Markdown instead of dropping them entirely
- **Semantic faithfulness verification** to reliably catch the hallucination class found above, replacing the abandoned regex-based attempt
- **Conversation memory** so follow-up questions can reference prior turns
- **Re-ranking step** to reduce the repetition observed in 3/10 test answers
- **Expanded document scope** once ingestion can run across multiple days without hitting free-tier quota limits
