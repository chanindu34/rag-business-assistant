import numpy as np
import pytest

from retriever import HybridRetriever

DOCS = [
    "group ebitda increased by 75 percent",
    "hotel room inventory in sri lanka",
    "risk management and cybersecurity",
    "supermarket expansion across the island",
]
# One-hot vectors: document i points along axis i.
EMB = np.eye(4).tolist()


@pytest.fixture
def hr():
    return HybridRetriever(DOCS, EMB, rrf_k=60)


def test_rrf_score_is_sum_of_reciprocal_ranks(hr):
    # Keyword match on doc 0 only; dense query aligned with doc 0.
    ids, scores = hr.retrieve_ids("ebitda", [1, 0, 0, 0], k=4)
    assert ids[0] == 0
    # Rank 1 in both lists: 1/(60+1) + 1/(60+1).
    assert scores[0] == pytest.approx(2 / 61)


def test_documents_without_keyword_overlap_get_no_bm25_credit(hr):
    ids, scores = hr.retrieve_ids("ebitda", [0, 1, 0, 0], k=4)
    by_id = dict(zip(ids.tolist(), scores.tolist()))
    # Doc 1: dense rank 1, no keyword overlap, so no BM25 term at all.
    assert by_id[1] == pytest.approx(1 / 61)
    # Doc 0: BM25 rank 1, dense rank 2 (ties broken by position).
    assert by_id[0] == pytest.approx(1 / 61 + 1 / 62)
    # Docs 2 and 3 share no keyword with the query: dense credit only.
    assert by_id[2] == pytest.approx(1 / 63) and by_id[3] == pytest.approx(1 / 64)


def test_zero_query_vector_falls_back_to_keywords(hr):
    chunks, _ = hr.retrieve("cybersecurity risk", [0, 0, 0, 0], k=1)
    assert chunks == [DOCS[2]]


def test_wrong_size_query_vector_does_not_crash(hr):
    chunks, _ = hr.retrieve("hotel rooms", [1.0] * 7, k=1)
    assert chunks == [DOCS[1]]


def test_none_query_vector_is_keyword_only(hr):
    chunks, _ = hr.retrieve("supermarket", None, k=1)
    assert chunks == [DOCS[3]]


def test_punctuation_only_query_uses_dense_only(hr):
    chunks, _ = hr.retrieve("???", [0, 0, 1, 0], k=1)
    assert chunks == [DOCS[2]]


def test_k_larger_than_corpus(hr):
    chunks, scores = hr.retrieve("group", [1, 0, 0, 0], k=50)
    assert len(chunks) == len(DOCS) == len(scores)


def test_zero_document_vector_is_tolerated():
    emb = [[0, 0], [1, 0]]
    hr = HybridRetriever(["a b", "c d"], emb)
    chunks, _ = hr.retrieve("zzz", [1, 0], k=1)
    assert chunks == ["c d"]
