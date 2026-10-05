import pytest

from pipeline import REFUSAL_TOKEN, build_prompt, peek_refusal, sources_from_result, validate_question


# ---------- input validation ----------
def test_whitespace_is_collapsed():
    assert validate_question("  How much\n\tdid EBITDA grow?  ") == ("How much did EBITDA grow?", None)


@pytest.mark.parametrize("raw", [None, "", "   \n\t "])
def test_empty_input_is_ignored(raw):
    assert validate_question(raw) == (None, None)


def test_overlong_input_is_rejected_before_any_api_call():
    q, err = validate_question("x" * 501, max_chars=500)
    assert q is None and "501" in err


def test_control_characters_removed():
    assert validate_question("EBITDA\x00\x07 growth")[0] == "EBITDA growth"


# ---------- refusal detection on a stream ----------
def _join(it):
    return "".join(it)


@pytest.mark.parametrize("pieces", [
    ["NOT_IN_REPORT"],
    ["NOT", "_IN", "_REP", "ORT"],          # token split across chunks
    ["  NOT_IN_", "REPORT\n"],              # surrounding whitespace
])
def test_refusal_detected(pieces):
    refused, rest = peek_refusal(pieces)
    assert refused and _join(rest) == ""


def test_normal_answer_streams_unchanged():
    pieces = ["Group EBITDA ", "increased by ", "75% [1]."]
    refused, rest = peek_refusal(pieces)
    assert not refused and _join(rest) == "".join(pieces)


def test_answer_starting_like_token_is_not_a_refusal():
    pieces = ["NOT", " all segments grew."]
    refused, rest = peek_refusal(pieces)
    assert not refused and _join(rest) == "NOT all segments grew."


def test_decision_made_after_first_piece_for_normal_answers():
    consumed = []

    def gen():
        for p in ["The ", "Group ", "manages ", "3,468 rooms."]:
            consumed.append(p)
            yield p

    refused, _ = peek_refusal(gen())
    assert not refused and consumed == ["The "]  # only one piece held back


def test_empty_stream():
    refused, rest = peek_refusal([])
    assert not refused and _join(rest) == ""


# ---------- sources and prompt ----------
def test_children_from_same_parent_are_sent_once():
    final = [
        {"chunk": "child a", "parent_id": "p1", "page": 10, "context": "PARENT ONE"},
        {"chunk": "child b", "parent_id": "p1", "page": 10, "context": "PARENT ONE"},
        {"chunk": "child c", "parent_id": "p2", "page": 36, "context": "PARENT TWO"},
    ]
    sources = sources_from_result(final)
    assert [s["text"] for s in sources] == ["PARENT ONE", "PARENT TWO"]
    assert sources[0]["match"] == "child a" and sources[1]["page"] == 36


def test_prompt_has_page_labels_refusal_contract_and_follow_up():
    prompt = build_prompt(
        "Why did it grow?",
        [{"page": 13, "text": "EBITDA increased by 75%."}],
        previous_question="How much did EBITDA grow?",
    )
    assert "[1] (page 13) EBITDA increased by 75%." in prompt
    assert REFUSAL_TOKEN in prompt
    assert 'follow-up to the earlier question: "How much did EBITDA grow?"' in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_user_text_shows_literally():
    from pipeline import escape_markdown
    assert escape_markdown("9**9**9") == r"9\*\*9\*\*9"
    assert escape_markdown("plain words") == "plain words"


def test_dollar_signs_do_not_render_as_maths_and_escaping_twice_is_safe():
    from pipeline import plain_answer
    assert plain_answer("between $5 and $6") == r"between \$5 and \$6"
    assert plain_answer(plain_answer("costs $5")) == r"costs \$5"


def test_maths_in_answers_stays_literal_but_bold_numbers_still_work():
    from pipeline import plain_answer
    assert plain_answer("9**9**9 is too large") == r"9\*\*9\*\*9 is too large"
    assert plain_answer("EBITDA was **Rs. 80.01 billion**") == "EBITDA was **Rs. 80.01 billion**"
    assert plain_answer("grew **75%** (page 14)") == "grew **75%** (page 14)"
