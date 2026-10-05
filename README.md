# Business Intelligence Assistant

[![tests](https://github.com/chanindu34/rag-business-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/chanindu34/rag-business-assistant/actions/workflows/tests.yml)

A retrieval-augmented generation (RAG) system that answers questions about John Keells Holdings' Annual Report 2025/26, citing the page each claim comes from.

**Live demo:** https://rag-business-assistant-pdpsjsuaxzvmaaff8edv3b.streamlit.app

## What it does

Ask a question in plain English. The app finds the relevant passages in the report, writes an answer using only those passages, cites them as [1], [2] with page numbers, and tells you how confident it is. If the report does not contain the answer, it says so instead of guessing.

## Architecture

```
Question
  -> Follow-up resolution     "Why did it grow?" becomes "How much did EBITDA grow? Why did it grow?"
  -> Semantic router          fact lookup: skip HyDE | explanation or change question: run HyDE
  -> HyDE (optional)          draft a hypothetical answer, embed the draft
  -> Hybrid search            BM25 (stemmed keywords) + dense (cosine), fused with Reciprocal Rank Fusion
  -> Cross-encoder rerank     30 candidates scored, top 8 kept
  -> Parent expansion         each matched 500-char passage is swapped for its 2,000-char parent
  -> Confidence tier          high | moderate (answer with a caution note) | junk filtered before the LLM
  -> Grounded generation      answer from context only, cited, streamed; NOT_IN_REPORT when unanswerable
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | Gemini `gemini-embedding-001`, 3072 dims, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types | HyDE drafts are embedded as documents, raw questions as queries |
| Sparse search | BM25 over a tokenizer that lowercases, strips punctuation and lightly stems | "manage" matches "management", "EBITDA?" matches "EBITDA" |
| Fusion | Weighted Reciprocal Rank Fusion, k = 60 | Ranks, not raw scores, so BM25 and cosine never need normalising |
| Reranker | `BAAI/bge-reranker-base` locally; `ms-marco-MiniLM-L-6-v2` on Streamlit Cloud | Ranked on raw logits; the default sigmoid output saturates at 1.00 and ties |
| Generation | Gemini flash models with an ordered fallback chain | Each model has its own free-tier quota; 503s and exhausted quotas fail over in under a second |
| Vector store | ChromaDB (persistent), brute-force cosine in NumPy | ~1,000 vectors does not need an ANN index |
| UI | Streamlit with live pipeline status, streaming answers and page-tagged sources | |

### Ingestion

- Text is extracted page by page so every chunk knows its page.
- Parent-child chunking: each page splits into parents (~2,000 chars, never crossing a page), each parent into children (~500 chars). Children are searched; the answer model receives parents.
- Embeddings are requested in batches of 100 and cached on disk by (model, task type, text). A run that stops on a quota error resumes for free, and rebuilding an unchanged index costs zero API calls.
- The index is built under a temporary name and swapped in only when complete, so a failed run never breaks the live index.

### Scope

Pages 1 to 60 (overview, Chairperson's message, management discussion, Group financial review) and 139 to 160 (outlook and risks, share information): 82 of 612 pages, 968 searchable passages. The industry group deep dives and the financial statements are excluded: the statements are mostly tables, and plain text extraction flattens a table into disconnected numbers.

## Resilience

- **Quota and outages:** answer and HyDE calls walk an ordered model chain. A daily quota error is never retried; a 503 gets one quick retry, then the next model. Every Gemini call has a 60 second timeout.
- **Embedding API down:** retrieval falls back to keyword search only, and the answer is labelled accordingly.
- **Repeat questions** are served from a disk cache with no API call. Refusals are not cached, so a retry gets a fresh attempt.
- **Input limits:** empty input is ignored; questions over 500 characters are rejected before any API call.

## What testing found

These came from testing the system on real questions, using `debug_retrieval.py` to trace where the correct passage ranks at each stage (BM25, dense, fused, reranked).

1. **The original index covered 16 pages, not 82.** The first ingestion embedded one chunk per API request and stopped on a quota error after 150 of 820 chunks. The app had only ever searched pages 1 to 16. Rebuilding with batched, cached embeddings took 10 requests.
2. **Vocabulary mismatch.** "How much did Group EBITDA grow?" failed because the report says "increased by 75%", never "grew". BM25 ranked the six passages containing the answer between 46th and 159th. The router had skipped HyDE because it saw a metric and a year, so questions about change now always run HyDE.
3. **Stemming.** For "How many hotel rooms does the Group manage?", the correct passage went from BM25 rank 16 to rank 1 once "manage" could match "management".
4. **Reranker scores do not measure answerability.** An off-topic question ("What is the capital of France?") scored 0.31, higher than a real one whose answer retrieval ranked first (0.29). No single threshold separates them. The reranker now only orders passages and filters obvious junk; the answer model decides whether the passages contain the answer and replies `NOT_IN_REPORT` if not.
5. **Parent context rescued a weak reranker.** For the EBITDA question the reranker placed the Group level passage 8th, below several industry group passages. A neighbouring child in the same parent made the top 6, so the model still received the answer. Sending 8 passages instead of 6 covers this case; a stronger reranker is the proper fix.
6. **The report contradicts itself.** Page 10 gives 3,468 rooms under management "as at 31 March 2026"; page 36 gives 3,577 with no date. The assistant reports both, with citations, rather than picking one.
7. **Prompt-only grounding has a limit.** An earlier version named an ESG initiative that appeared in none of the retrieved passages. Instructions alone did not stop it, which is why the system now has an explicit refusal path and why claim-level faithfulness checking is on the list below.

## Evaluation results

Run on 5 October 2026 with `python3 evaluate.py --judge`.

| Metric | Result |
| --- | --- |
| Retrieval: a correct page reached the answer model | 24 / 24 |
| Answer accuracy: every expected figure present | 24 / 24 |
| Unanswerable questions correctly declined | 4 / 4 |
| Answerable questions wrongly declined | 0 / 24 |
| Faithfulness: the judge found no unsupported claim | 24 / 24 |
| End to end latency, median | 12.2 s |

How to read these numbers:

- **The raw report said 23 / 24.** The one failure was the evaluator, not the system: it answered "Rs. 35,723 million" and the key expected "35.72" (billion). The key now accepts both. Failures are checked by hand before a metric is trusted.
- **Reranker scores confirm finding 4.** Answerable questions scored 0.51 to 1.00; unanswerable ones scored 0.02, 0.10, 0.62 and 1.00. Two out of scope questions scored as high as real ones, so no threshold could separate them. All four were declined by the answer model's `NOT_IN_REPORT` decision.
- **28 questions is a small set.** At 24 / 24 the 95 percent interval for accuracy is still roughly 86 to 100 percent. It catches regressions; it cannot rank two close configurations.
- **Latency p95 was 127 s** because Gemini returned 503 and 504 errors and some models ran out of daily quota during the run, so requests moved down the model chain. The median is the representative figure.
- **Not every answer came from the first choice model.** gemini-3.8-flash ran out of quota part way through; the run does not yet record which model produced each answer.

## Known limitations

- **Coverage:** 82 of 612 pages. Tables are flattened, so questions about the financial statements will be declined.
- **Reranker:** `bge-reranker-base` tends to rank industry group figures above Group level ones. `BAAI/bge-reranker-v2-m3` should do better but is twice the size; it is a one-line change in `config.yaml` and measurable with `debug_retrieval.py`.
- **Confidence thresholds** (0.6 and 0.05) only filter obvious junk. The evaluation shows reranker scores cannot separate answerable from unanswerable questions, so the answer model makes that call.
- **Follow-up detection** is a rule-based heuristic. An LLM rewrite would handle more phrasings at the cost of one API call per follow-up.
- **Not multi-tenant:** no authentication, one Streamlit process, local Chroma files. Fine for a demo, not for production traffic.

## Tech stack

Python, Google Gemini API, ChromaDB, rank-bm25, sentence-transformers (cross-encoder), LangChain text splitters, pypdf, Streamlit, Docker

## Project structure

```
app.py                 Streamlit UI: chat, live pipeline status, streaming, sources
pipeline.py            Builds the pipeline without any UI; shared by the app, tests and evaluation
retriever.py           Router, HyDE, hybrid search with RRF, reranker, confidence tiers
semantic_router.py     Decides when HyDE is worth an LLM call
query_condensation.py  Resolves follow-up questions without an API call
confidence_tiers.py    High / moderate / low tiers on the reranker score
llm.py                 Gemini gateway: model fallback, quota handling, streaming, answer cache
ingest.py              Builds the index: page extraction, parent-child chunks, batched cached embeddings
embedding_cache.py     On-disk embedding cache
evaluate.py            Runs the evaluation set and writes a scored report to eval/results/
eval/questions.yaml    28 questions with expected figures and source pages, checked against the report
debug_retrieval.py     Shows where the correct passage ranks at every stage
list_models.py         Lists the Gemini models your API key can use
config.yaml            Every tunable setting, with the reasoning behind non-obvious values
chroma_db/             Prebuilt index (committed)
tests/                 88 unit tests: tokenizer, RRF, router, follow-ups, refusal detection, model failover
experimental/          Earlier modules not used by the app, with notes on why
```

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
streamlit run app.py
```

The first start downloads the reranker (about 1.1 GB). After that, setting `HF_HUB_OFFLINE=1` in `.env` skips the update check, which helps on networks that block Hugging Face.

Run `python3 list_models.py` to see which Gemini models your key can use, and edit the model chain in `config.yaml` to match.

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests
```

The tests use fake Gemini clients and a fake reranker: no API key, no network, no model download, under a second. They run on every push via GitHub Actions. Each one guards a behaviour that was once broken or easy to break, for example stemming ("manage" must match "management"), the HyDE rule for change questions, never retrying a daily quota error, and detecting a `NOT_IN_REPORT` refusal even when it streams in pieces.

### Evaluation

```bash
python3 evaluate.py --check     # verify every expected answer appears on its listed page (no API)
python3 evaluate.py --judge     # full run, about 3 API calls per question
```

28 questions with answers and page numbers taken from the report (`eval/questions.yaml`): lookups, change questions, explanations, follow ups and four questions the report cannot answer. `--check` verifies the answer key against the indexed text without any API call. The judge is a different model from the one that answers. Results are in [Evaluation results](#evaluation-results).

### Docker

```bash
docker build -t rag-assistant .
docker run -p 8501:8501 --env-file .env rag-assistant
```

The image uses CPU-only PyTorch, bakes the reranker in at build time and runs as a non-root user with a health check on Streamlit's `/_stcore/health`. For a smaller image: `--build-arg RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Rebuilding the index

1. Save the annual report PDF as `data/Annual Report.pdf` (not committed because of its size).
2. Run:

```bash
pip install -r requirements-ingest.txt
python3 ingest.py --dry-run     # page and chunk counts, no API calls
python3 ingest.py               # about 10 embedding requests
```

## What I'd build next

- **A larger evaluation set** (100 or more questions) run on a schedule, recording which model answered each question, so two configurations can be compared fairly.
- **Claim-level faithfulness checking:** verify each cited sentence against its source passage before showing the answer.
- **Layout-aware parsing** (for example LlamaParse or Docling) to bring the financial statement tables in with their structure intact.
- **A stronger reranker**, evaluated against the current one with `debug_retrieval.py`.
