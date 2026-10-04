from unittest.mock import MagicMock

import pytest
from google.genai import errors

import llm
from llm import GeminiGateway, QuotaExhaustedError

DAILY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                   "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}
OVERLOADED = {"error": {"code": 503, "status": "UNAVAILABLE", "message": "overloaded"}}
NOT_FOUND = {"error": {"code": 404, "status": "NOT_FOUND", "message": "no such model"}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def client_with(behaviour):
    """behaviour: model -> exception instance or answer text."""
    calls = []

    def respond(model, **_):
        calls.append(model)
        b = behaviour[model]
        if isinstance(b, Exception):
            raise b
        return MagicMock(text=b)

    def stream(model, **_):
        calls.append(model)
        b = behaviour[model]
        if isinstance(b, Exception):
            raise b
        return iter([MagicMock(text=w) for w in b.split("|")])

    c = MagicMock()
    c.models.generate_content.side_effect = respond
    c.models.generate_content_stream.side_effect = stream
    return c, calls


def test_daily_quota_moves_to_next_model_without_retrying():
    c, calls = client_with({"a": errors.ClientError(429, DAILY), "b": "answer"})
    assert GeminiGateway(c, ["a", "b"]).generate("q") == "answer"
    assert calls == ["a", "b"]


def test_overloaded_model_gets_one_retry_then_fails_over():
    c, calls = client_with({"a": errors.ServerError(503, OVERLOADED), "b": "answer"})
    assert GeminiGateway(c, ["a", "b"]).generate("q") == "answer"
    assert calls == ["a", "a", "b"]


def test_missing_model_is_skipped_and_reported():
    c, _ = client_with({"a": errors.ClientError(404, NOT_FOUND), "b": errors.ClientError(429, DAILY)})
    with pytest.raises(QuotaExhaustedError, match="not available on this API key: a"):
        GeminiGateway(c, ["a", "b"]).generate("q")


def test_exhausted_model_is_skipped_on_later_questions():
    c, calls = client_with({"a": errors.ClientError(429, DAILY), "b": "answer"})
    g = GeminiGateway(c, ["a", "b"])
    g.generate("q1")
    g.generate("q2")
    assert calls == ["a", "b", "b"]


def test_cache_hit_makes_no_api_call(tmp_path):
    c, calls = client_with({"a": "answer"})
    g = GeminiGateway(c, ["a"], cache_path=str(tmp_path / "cache.json"))
    g.generate("same question")
    g.generate("same question")
    assert calls == ["a"]


def test_refusals_are_not_cached(tmp_path):
    c, calls = client_with({"a": "NOT_IN_REPORT"})
    g = GeminiGateway(c, ["a"], cache_path=str(tmp_path / "c.json"), never_cache={"NOT_IN_REPORT"})
    list(g.stream("q"))
    list(g.stream("q"))
    assert calls == ["a", "a"]


def test_stream_fails_over_before_first_piece():
    c, calls = client_with({"a": errors.ServerError(503, OVERLOADED), "b": "Group |EBITDA |rose"})
    assert "".join(GeminiGateway(c, ["a", "b"]).stream("q")) == "Group EBITDA rose"
    assert calls == ["a", "a", "b"]


def test_stream_raises_when_every_model_is_out():
    c, _ = client_with({"a": errors.ClientError(429, DAILY)})
    with pytest.raises(QuotaExhaustedError):
        list(GeminiGateway(c, ["a"]).stream("q"))


def test_other_client_errors_are_not_retried():
    c, calls = client_with({"a": errors.ClientError(400, {"error": {"code": 400, "message": "bad"}}), "b": "x"})
    with pytest.raises(errors.ClientError):
        GeminiGateway(c, ["a", "b"]).generate("q")
    assert calls == ["a"]
