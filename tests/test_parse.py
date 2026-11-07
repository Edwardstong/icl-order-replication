import pytest

from iclorder.parse import cut_at_stop, extract_number, gsm8k_gold, numeric_match


def test_gold_from_reference():
    assert gsm8k_gold("She has 3 + 4 = 7 apples.\n#### 7") == "7"
    assert gsm8k_gold("...\n#### 1,250") == "1250"
    with pytest.raises(ValueError):
        gsm8k_gold("no final line")


def test_extract_prefers_hash_then_last_number():
    assert extract_number("so 2 + 2 = 4\n#### 4") == 4.0
    assert extract_number("first 10 then 25 then the answer is 35.") == 35.0
    assert extract_number("The total is $1,200.") == 1200.0
    assert extract_number("no digits here") is None


def test_cut_at_stop_removes_hallucinated_next_question():
    text = " 7\n\nQuestion: next one\nAnswer: 9"
    assert cut_at_stop(text) == " 7"


def test_numeric_match_tolerance():
    assert numeric_match(7.0, "7")
    assert numeric_match(1000.05, "1000")        # within 1e-4 relative
    assert not numeric_match(1001.0, "1000")
    assert not numeric_match(None, "7")
    assert numeric_match(0.0, "0")
