"""Show where the passage that should answer a question ranks at every stage.

    python3 debug_retrieval.py "How many hotel rooms does the Group manage?" "3,468 rooms"

Arguments: the question, then a short phrase that appears in the correct
passage. Costs 1 embedding call (plus 1 HyDE call if the router picks HyDE).
"""
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.WARNING)

import chromadb
from google import genai
from google.genai import types

from config import (CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, HYDE_MODEL, NUM_CANDIDATES,
                    RERANKER_MODEL, RRF_K, parents_path)
from retriever import HybridRetriever, RerankerFilter, _ranks, tokenize
from semantic_router import SemanticRouter


def rank_of(scores, idx):
    return int((np.asarray(scores) > scores[idx]).sum()) + 1


def main(question, phrase):
    col = chromadb.PersistentClient(path=CHROMA_DB_PATH).get_collection(COLLECTION_NAME)
    r = col.get(include=["documents", "embeddings", "metadatas"])
    docs, metas = r["documents"], r["metadatas"]
    gold = [i for i, d in enumerate(docs) if phrase.lower() in d.lower()]
    if not gold:
        print(f'No chunk contains "{phrase}". Try a shorter phrase.')
        return
    print(f'{len(gold)} chunk(s) contain "{phrase}": pages {[metas[i]["page"] for i in gold]}')

    hr = HybridRetriever(docs, r["embeddings"], rrf_k=RRF_K)
    route = SemanticRouter().classify(question)
    client = genai.Client()
    dense_text, task = question, "RETRIEVAL_QUERY"
    if route == "run_hyde":
        dense_text = client.models.generate_content(model=HYDE_MODEL, contents=(
            f"Generate a plausible, detailed answer to this question.\nBe specific, as if answering based on a "
            f"document.\nOnly generate the answer, no preamble.\n\nQuestion: {question}\n\nAnswer:")).text
        task = "RETRIEVAL_DOCUMENT"
    q = np.asarray(client.models.embed_content(model=EMBEDDING_MODEL, contents=dense_text,
                   config=types.EmbedContentConfig(task_type=task)).embeddings[0].values, dtype=np.float32)

    bm25 = np.asarray(hr.bm25.get_scores(tokenize(question)))
    dense = hr.embeddings @ (q / np.linalg.norm(q))
    cand, _ = hr.retrieve_ids(question, q, k=NUM_CANDIDATES)
    print(f"\nRouter: {route}   BM25 tokens: {tokenize(question)}")
    print(f"{'chunk':>6} {'page':>5} {'BM25':>6} {'dense':>6} {'fused':>8}")
    fused_rank = {int(i): n + 1 for n, i in enumerate(cand)}
    for g in gold:
        fr = fused_rank.get(g, f">{NUM_CANDIDATES}")
        print(f"{g:>6} {metas[g]['page']:>5} {rank_of(bm25, g):>6} {rank_of(dense, g):>6} {str(fr):>8}")

    rr = RerankerFilter(RERANKER_MODEL)
    out, _ = rr.rerank(question, [docs[i] for i in cand], top_k=len(cand))
    print("\nReranked top 8 (* = contains the phrase):")
    for o in out[:8]:
        i = int(cand[o["pos"]])
        mark = "*" if i in gold else " "
        print(f" {mark} p{metas[i]['page']:<4} logit {o['rerank_logit']:6.2f}  score {o['rerank_score']:.2f}  {docs[i][:70]!r}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
