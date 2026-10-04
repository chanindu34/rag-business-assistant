"""ProductionRAG end to end with a fake Gemini client and a fake reranker:
no network, no model download."""
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.genai import errors

import retriever
from retriever import ProductionRAG

DOCS = [
    "Combined room inventory of 3,468 rooms under management in Sri Lanka and the Maldives",
    "Group EBITDA increased by 75% to Rs.80.01 billion",
    "Key risks include cybersecurity and macroeconomic volatility",
]
METAS = [{"page": 10, "parent_id": "p10"}, {"page": 13, "parent_id": "p13"}, {"page": 140, "parent_id": "p140"}]
PARENTS = {"p10": {"page": 10, "text": "PARENT 10"}, "p13": {"page": 13, "text": "PARENT 13"},
           "p140": {"page": 140, "text": "PARENT 140"}}


class FakeReranker:
    """Scores by word overlap; score_cap lets a test force a confidence tier."""
    score_cap = 1.0

    def __init__(self, name):
        pass

    def warm_up(self):
        pass

    def rerank(self, query, candidates, top_k=3):
        q = set(query.lower().split())
        scores = [min(len(q & set(c.lower().split())) / max(len(q), 1), self.score_cap) for c in candidates]
        order = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": candidates[i], "rerank_score": float(scores[i]), "rank": j, "pos": int(i)}
                for j, i in enumerate(order)], 1.0


@pytest.fixture
def make_rag(monkeypatch):
    monkeypatch.setattr(retriever, "RerankerFilter", FakeReranker)
    monkeypatch.setattr(retriever.time, "sleep", lambda s: None)

    def build(embed_error=None, score_cap=1.0, hyde_text="draft answer"):
        FakeReranker.score_cap = score_cap
        client = MagicMock()
        if embed_error:
            client.models.embed_content.side_effect = embed_error
        else:
            client.models.embed_content.return_value.embeddings = [MagicMock(values=[1.0, 0.0, 0.0])]
        rag = ProductionRAG(
            DOCS, np.eye(3).tolist(), "emb", "gen", client,
            hyde_generate_fn=lambda prompt: hyde_text,
            metadatas=METAS, parents=PARENTS, high_threshold=0.6, low_threshold=0.05,
        )
        return rag, client
    return build


def test_fact_lookup_skips_hyde_and_returns_parent_with_page(make_rag):
    rag, _ = make_rag()
    steps = []
    out = rag.retrieve("How many rooms under management?", top_k=1, on_step=steps.append)
    top = out["final_chunks"][0]
    assert out["retrieval_stats"]["route"] == "skip_hyde" and not out["retrieval_stats"]["hyde_used"]
    assert top["page"] == 10 and top["context"] == "PARENT 10"
    assert steps[0].startswith("Fact lookup") and steps[-1].startswith("Reranking")


def test_change_question_runs_hyde(make_rag):
    rag, _ = make_rag()
    out = rag.retrieve("How much did Group EBITDA grow?", top_k=1)
    assert out["retrieval_stats"]["hyde_used"] and out["hyde_generated"] == "draft answer"


def test_low_tier_returns_no_chunks(make_rag):
    rag, _ = make_rag(score_cap=0.01)
    out = rag.retrieve("What is the capital of France?", top_k=2)
    assert out["confidence"] == "low" and out["final_chunks"] == []


def test_ambiguous_tier_keeps_chunks(make_rag):
    rag, _ = make_rag(score_cap=0.3)
    out = rag.retrieve("Key risks include cybersecurity?", top_k=2)
    assert out["confidence"] == "ambiguous" and out["final_chunks"]


def test_embedding_outage_falls_back_to_keyword_search(make_rag):
    rag, client = make_rag(embed_error=errors.ServerError(503, {"error": {"code": 503, "message": "down"}}))
    out = rag.retrieve("How many rooms under management?", top_k=1)
    assert out["retrieval_stats"]["dense_unavailable"]
    assert out["final_chunks"][0]["page"] == 10
    assert client.models.embed_content.call_count == 3  # retried, then gave up


def test_daily_embedding_quota_is_not_retried(make_rag):
    daily = errors.ClientError(429, {"error": {"code": 429, "message": "quota",
                                               "details": [{"quotaId": "EmbedPerDay"}]}})
    rag, client = make_rag(embed_error=daily)
    rag.retrieve("rooms", top_k=1)
    assert client.models.embed_content.call_count == 1
