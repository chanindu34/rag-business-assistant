"""
Build the vector index for John Keells Holdings' Annual Report 2025/26.

    python3 ingest.py --dry-run   # page and chunk counts, no API calls
    python3 ingest.py             # embed and build the index

Pipeline
1. Extract text page by page (page numbers are kept for citations).
2. Parent-child chunking. Each page is split into parents (~2000 chars,
   never crossing a page boundary), and each parent into children
   (~500 chars). Children are embedded and searched for precision; the
   answer model receives the parent for context.
3. Embed children in batches of up to 100 texts per API request, with the
   RETRIEVAL_DOCUMENT task type. Every vector is cached on disk, so a run
   that stops on a quota error resumes for free.
4. Build into "<collection>__building" and only swap it in once complete,
   so a failed run never touches the live index.

Previous version (kept in git history) embedded one chunk per request and
had stopped after 150 of ~800 chunks, so the live index covered pages 1-16.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL,
    PAGE_RANGES,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PDF_PATH,
    parents_path,
)
from embedding_cache import EmbeddingCache

DOC_TASK_TYPE = "RETRIEVAL_DOCUMENT"

# pypdf warns once per font that it could not fully decode without fontTools.
# Text extraction still works, so keep the console readable.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


def load_pages(path: str):
    reader = PdfReader(path)
    pages, failed = [], []
    for start, end in PAGE_RANGES:
        for i in range(start, min(end, len(reader.pages))):
            try:
                text = clean(reader.pages[i].extract_text() or "")
            except Exception as e:  # one corrupt page must not kill the run
                failed.append((i + 1, str(e)[:80]))
                continue
            if len(text) >= 50:
                pages.append({"page": i + 1, "text": text})  # 1-based PDF page number
    return pages, failed, len(reader.pages)


def chunk(pages):
    parent_split = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
    child_split = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
    parents, children = {}, []
    for p in pages:
        for pi, parent_text in enumerate(parent_split.split_text(p["text"])):
            parent_id = f"p{p['page']:04d}_{pi:02d}"
            parents[parent_id] = {"page": p["page"], "text": parent_text}
            for ci, child_text in enumerate(child_split.split_text(parent_text)):
                children.append({
                    "id": f"{parent_id}_c{ci:02d}",
                    "text": child_text,
                    "metadata": {"page": p["page"], "parent_id": parent_id},
                })
    return parents, children


def _daily_quota(err) -> bool:
    return "PerDay" in str(err) or "per day" in str(err).lower()


def embed_all(children, cache: EmbeddingCache):
    from google import genai
    from google.genai import errors, types

    keys = [EmbeddingCache.key(EMBEDDING_MODEL, DOC_TASK_TYPE, c["text"]) for c in children]
    cached = cache.get_many(keys)
    todo = [i for i, k in enumerate(keys) if k not in cached]
    print(f"Embeddings: {len(cached)} cached, {len(todo)} to embed "
          f"(~{-(-len(todo) // EMBED_BATCH_SIZE)} API requests).")

    client = genai.Client()
    config = types.EmbedContentConfig(task_type=DOC_TASK_TYPE)
    for b in range(0, len(todo), EMBED_BATCH_SIZE):
        batch = todo[b:b + EMBED_BATCH_SIZE]
        for attempt in range(6):
            try:
                resp = client.models.embed_content(
                    model=EMBEDDING_MODEL, contents=[children[i]["text"] for i in batch], config=config
                )
                break
            except errors.ClientError as e:
                if e.code == 429 and _daily_quota(e):
                    print(f"\nDaily embedding quota reached after {b} of {len(todo)} new chunks.")
                    print("Progress is cached. Re-run `python3 ingest.py` after the quota resets.")
                    print("The live index was NOT changed.")
                    sys.exit(2)
                if e.code == 429 and attempt < 5:
                    m = re.search(r"retry in ([\d.]+)s", str(e))
                    wait = min(float(m.group(1)) if m else 2 ** attempt * 5, 90) + random.uniform(0, 2)
                    print(f"  rate limited, waiting {wait:.0f}s ...")
                    time.sleep(wait)
                    continue
                raise
            except errors.ServerError:
                if attempt < 5:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                raise
        vectors = [e.values for e in resp.embeddings]
        if len(vectors) != len(batch):
            raise RuntimeError(f"API returned {len(vectors)} vectors for {len(batch)} texts")
        new = {keys[i]: v for i, v in zip(batch, vectors)}
        cache.put_many(new)
        cached.update(new)
        print(f"  embedded {min(b + EMBED_BATCH_SIZE, len(todo))}/{len(todo)}")

    vectors = [cached[k] for k in keys]
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"Mixed embedding sizes {dims}; refusing to build a corrupt index.")
    zero = sum(1 for v in vectors if not any(v))
    if zero:
        raise RuntimeError(f"{zero} zero vectors returned; refusing to build the index.")
    return vectors


def build_collection(children, vectors, parents, n_pages_total):
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    building = f"{COLLECTION_NAME}__building"
    try:
        client.delete_collection(building)
    except Exception:
        pass
    col = client.create_collection(
        name=building,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": EMBEDDING_MODEL,
            "doc_task_type": DOC_TASK_TYPE,
            "child_chunk": f"{CHILD_CHUNK_SIZE}/{CHILD_CHUNK_OVERLAP}",
            "parent_chunk": f"{PARENT_CHUNK_SIZE}/{PARENT_CHUNK_OVERLAP}",
            "page_ranges": json.dumps(PAGE_RANGES),
            "built_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    for i in range(0, len(children), 500):
        part = children[i:i + 500]
        col.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[c["metadata"] for c in part],
            embeddings=vectors[i:i + 500],
        )
    if col.count() != len(children):
        raise RuntimeError(f"Stored {col.count()} of {len(children)} chunks; live index left unchanged.")

    # Parents first, then the atomic-enough swap of the collection name.
    path = Path(parents_path(COLLECTION_NAME))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(parents), encoding="utf-8")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col.modify(name=COLLECTION_NAME)
    tmp.replace(path)
    print(f"\nIndex '{COLLECTION_NAME}' built: {len(children)} chunks, {len(parents)} parents.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show page and chunk counts, no API calls")
    args = ap.parse_args()

    pages, failed, total = load_pages(PDF_PATH)
    parents, children = chunk(pages)
    print(f"PDF: {total} pages. Page ranges {PAGE_RANGES}: {len(pages)} pages with text.")
    if failed:
        print(f"Skipped {len(failed)} unreadable pages: {failed[:5]}")
    print(f"Chunks: {len(parents)} parents, {len(children)} children "
          f"({sum(len(c['text']) for c in children):,} chars).")
    if args.dry_run:
        cache = EmbeddingCache(EMBEDDING_CACHE_PATH)
        keys = [EmbeddingCache.key(EMBEDDING_MODEL, DOC_TASK_TYPE, c["text"]) for c in children]
        todo = len(keys) - len(cache.get_many(keys))
        print(f"Dry run: {todo} chunks need embedding (~{-(-todo // EMBED_BATCH_SIZE)} API requests). Nothing changed.")
        return

    vectors = embed_all(children, EmbeddingCache(EMBEDDING_CACHE_PATH))
    build_collection(children, vectors, parents, total)


if __name__ == "__main__":
    main()
