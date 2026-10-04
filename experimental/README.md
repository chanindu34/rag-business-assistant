# Experimental modules (not used by the app)

Kept for reference. None of these are imported by `app.py`, `retriever.py` or `ingest.py`.

| File | Status |
|---|---|
| `parent_child_metadata.py` | Superseded. Parent-child chunking now lives in `ingest.py` (page -> parent -> child, with `parent_id` metadata). |
| `resilient_embedding.py` | Superseded by batched, cached embedding in `ingest.py` + `embedding_cache.py`. Its local-model fallback mixed 384-dim vectors into a 3072-dim index, which corrupts search. |
| `bm25_persistence.py` | Not needed at this size (BM25 builds in under a second for ~1,000 chunks). Used `pickle.load` and Python's per-process `hash()`, so it would always report the index as stale. |
| `index_versioning.py` | Idea kept. Ingestion now builds into `<collection>__building` and swaps on success, which covers the main need. |
| `shadow_query_logging.py` | Idea kept for production: logging declined questions shows what the index is missing. Not wired in. |
| `Dockerfile.production` | Superseded by the root `Dockerfile`. This one could not start: it did not copy `config.yaml` or the index. |
