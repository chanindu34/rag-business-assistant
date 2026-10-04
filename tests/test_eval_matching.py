"""The evaluation is only as good as its matcher: these pin down its rules."""
import pytest

from evaluate import contains, expected_found


@pytest.mark.parametrize("text,needle", [
    ("increased by 75% to Rs.80.01 billion", "80.01"),
    ("Rs. 80.01 billion", "80.01"),
    ("recurring PBT of Rs.35.72 billion", "35.72"),
    ("3,468 rooms", "3,468"),
    ("3468 rooms", "3,468"),
    ("grew by 75 per cent", "75%"),
    ("GDP growth of 5% in 2025", "5%"),
    ("a **190%** increase", "190%"),
    ("Rs.500 million", "500 million"),
    ("Cybersecurity risks", "cybersecurity"),
])
def test_matches(text, needle):
    assert contains(text, needle)


@pytest.mark.parametrize("text,needle", [
    ("inflation of 5.4%", "5%"),        # must not match inside a different number
    ("an increase of 35%", "5%"),
    ("Rs.180.01 billion", "80.01"),
    ("Rs.80.012 billion", "80.01"),
    ("800 rooms", "80"),
])
def test_does_not_match_a_different_number(text, needle):
    assert not contains(text, needle)


def test_expected_found_reports_what_is_missing():
    q = {"must_include": ["75%", "80.01"], "any_of": ["Group", "Company"]}
    assert expected_found("Group EBITDA rose 75% to Rs.80.01 billion", q) == (True, [])
    ok, missing = expected_found("EBITDA rose 75%", q)
    assert not ok and "80.01" in missing and any("one of" in m for m in missing)
