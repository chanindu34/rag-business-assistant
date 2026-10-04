import pytest

from confidence_tiers import ConfidenceTier, ConfidenceTierRouter
from query_condensation import QueryCondenser
from semantic_router import SemanticRouter


@pytest.mark.parametrize("question,route", [
    ("How many hotel rooms does the Group manage?", "skip_hyde"),
    ("What was the Group's recurring profit before tax this year?", "skip_hyde"),
    ("What is the dividend per share?", "skip_hyde"),
    # Change questions: the report says "increased", never "grew".
    ("How much did Group EBITDA grow in 2025/26?", "run_hyde"),
    ("How did Retail perform?", "run_hyde"),
    ("What are the key risks facing the Group?", "run_hyde"),
    ("Why did EBITDA rise 75%?", "run_hyde"),
    ("What is the capital of France?", "run_hyde"),
])
def test_router(question, route):
    assert SemanticRouter().classify(question) == route


@pytest.mark.parametrize("question,is_follow_up", [
    ("Why did it grow so much?", True),
    ("What drove that?", True),
    ("And the retail segment?", True),
    ("What about Transportation?", True),
    ("How did that segment perform?", True),
    ("Compare its margin to last year", True),
    # Must NOT be treated as follow-ups:
    ("What was the Group's recurring profit before tax this year?", False),
    ("What are the risks that the company faces?", False),
    ("What is the profit for this year?", False),
])
def test_follow_up_detection(question, is_follow_up):
    assert QueryCondenser().is_follow_up(question) is is_follow_up


def test_follow_ups_chain_through_resolved_queries():
    c, history = QueryCondenser(), []
    for q in ["How much did Group EBITDA grow?", "Why did it grow so much?", "What about Transportation?"]:
        resolved, _ = c.resolve(history, q)
        history.append({"role": "user", "content": q, "resolved_query": resolved})
    assert history[-1]["resolved_query"].startswith("How much did Group EBITDA grow?")
    assert history[-1]["resolved_query"].endswith("What about Transportation?")


def test_standalone_question_is_not_rewritten():
    history = [{"role": "user", "content": "How much did EBITDA grow?"}]
    assert QueryCondenser().resolve(history, "What are the key risks?") == ("What are the key risks?", None)


def test_chained_context_is_capped():
    history = [{"role": "user", "content": "word " * 500}]
    resolved, _ = QueryCondenser().resolve(history, "Why did it change?")
    assert len(resolved.split()) <= 60 + len("Why did it change?".split())


@pytest.mark.parametrize("score,tier", [
    (0.95, ConfidenceTier.HIGH), (0.6, ConfidenceTier.HIGH),
    (0.59, ConfidenceTier.AMBIGUOUS), (0.05, ConfidenceTier.AMBIGUOUS),
    (0.049, ConfidenceTier.LOW), (0.0, ConfidenceTier.LOW),
])
def test_confidence_tiers(score, tier):
    assert ConfidenceTierRouter(high_threshold=0.6, low_threshold=0.05).classify_score(score) is tier
