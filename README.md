# Business Intelligence Assistant

A retrieval-augmented generation (RAG) system that answers natural-language questions about John Keells Holdings' Annual Report 2025/26, with source citations for every claim.

**Live demo:** https://rag-business-assistant-pdpsjsuaxzvmaaff8edv3b.streamlit.app

## What it does

A chat interface over a real 612-page annual report. Ask a question, get an answer synthesized from the actual document, not a general-knowledge guess, with numbered citations you can click to verify against the exact source text.

## Architecture

```
PDF → chunk (500 chars, 50 overlap) → embed (Gemini) → store (ChromaDB)
                                                              ↓
query → embed → retrieve top-k → build cited prompt → generate answer
```

- **Embeddings:** Gemini `gemini-embedding-001`, via a custom adapter class decoupling the vector store from any one provider
- **Vector store:** ChromaDB, persistent, with resumable batch ingestion (survives API failures mid-run without losing progress)
- **Generation:** Gemini `gemini-2.5-flash`, with exponential-backoff retry logic on both embedding and generation calls
- **Chunking:** LangChain's RecursiveCharacterTextSplitter. Chunk size chosen empirically (tested at 200/500/1000 chars; 500 was the only size that avoided both mid-sentence cuts and loss of retrieval precision)
- **Document scope:** deliberately limited to pages 0-60 and 138-160 (core financial/strategic narrative + outlook). Two reasons: pages 166-293 repeat Group-level facts at finer industry-group granularity, and the regulatory disclosure sections (294+) are structurally tabular (Topic, Metric Code, Unit of Measure). Standard text extraction flattens tables into disconnected values, losing the relational structure that gives a table its meaning. Scoping to narrative-heavy sections also kept ingestion within the embedding API's daily free-tier quota.

## Testing results

10 real questions, manually graded against the actual source text (not assumed correct):

- **9/10 answered correctly**, including hard synthesis questions with zero keyword overlap with the source phrasing
- **3/10 correct but repetitive**: multiple retrieved chunks restating the same point in slightly different words
- **1/10 contained a confirmed hallucination**: cited a specific initiative name that was verifiably absent from all retrieved context

### The hallucination, and what I learned from it

Asked about ESG initiatives, the model referenced a specific program by name that wasn't present in any of the 6 retrieved chunks, confirmed via direct string search, not assumption. I tried two fixes:

1. **Explicit prompt instruction** forbidding outside knowledge, even when the model recognizes the company. Did not resolve it on retest.
2. **A regex-based faithfulness check** flagging capitalized terms absent from retrieved context. Produced false positives on ordinary capitalized words while still missing the actual hallucinated term.

**Conclusion:** this is a known, real limitation of prompt-only grounding. LLMs can surface training-data knowledge about well-known public entities regardless of instructions. A production fix would need semantic-level faithfulness verification (comparing claim embeddings against source embeddings), not prompt engineering or string matching. I removed the unreliable regex check from the live app rather than ship something that produces misleading signals, documenting the finding here instead.

### Known limitation: no multi-turn memory

Each question is answered independently, retrieval and generation don't incorporate prior conversation turns. Follow-up questions using pronouns or references ("that one", "the other option") can't be resolved, and the system correctly says so rather than guessing. A production version would need to include recent conversation history in the retrieval/generation context.

## Tech stack

Python, Google Gemini API, ChromaDB, LangChain (text splitting), Streamlit, Docker

## Project structure

```
app.py            Streamlit chat app (retrieval, prompt building, generation)
ingest.py         One-time script that builds the vector store
config.py         Loads config.yaml and exposes settings to every script
config.yaml       Models, retrieval, chunking and rate limit settings
chroma_db/        Persistent ChromaDB vector store (committed)
data/             Source PDF goes here (not committed)
.env.example      Template for API keys
```

## Configuration

All tunable settings live in one place, `config.yaml`: model names, collection
name, top-k, chunk size and overlap, page ranges, retry and rate limit values.
Every script reads them through `config.py`, so the embedding model used to
build the index cannot drift from the one used to query it.

Any value can be overridden with an environment variable named
`RAG_<SECTION>_<KEY>`, for example `RAG_RETRIEVAL_TOP_K=8`. Secrets (API keys)
are never stored in `config.yaml`; they stay in `.env` or in the hosting
platform's secrets.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your real GEMINI_API_KEY in .env
export GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

Or with Docker:

```bash
docker build -t rag-assistant .
docker run -p 8501:8501 --env-file .env rag-assistant
```

## Rebuilding the vector store

`ingest.py` is the one-time script that chunked the source annual report with
LangChain's RecursiveCharacterTextSplitter and built the committed `chroma_db/`
store. It is not part of the running app and does not need to be re-run to use
the assistant. To re-run it:

1. Download the John Keells Holdings Annual Report 2025/26 PDF from the
   company's investor relations page.
2. Save it as `data/Annual Report.pdf` (the path is set by `data.pdf_path` in
   `config.yaml`; the PDF is not committed due to its size).
3. Run:

```bash
pip install -r requirements-ingest.txt
export GEMINI_API_KEY="your-key-here"
python ingest.py
```

Ingestion is resumable: if it stops partway, re-running continues from the last
stored batch. If you change the embedding model or chunking settings, delete
`chroma_db/` first so the whole index is rebuilt consistently.

## What I'd build next

- **Layout-aware document parsing** (e.g. LlamaParse) to properly handle the tabular sections that are currently excluded, preserving table structure as Markdown instead of dropping them entirely
- **Semantic faithfulness verification** to reliably catch the hallucination class found above, replacing the abandoned regex-based attempt
- **Conversation memory** so follow-up questions can reference prior turns
- **Re-ranking step** to reduce the repetition observed in 3/10 test answers
- **Expanded document scope** once ingestion can run across multiple days without hitting free-tier quota limits
