from retriever import tokenize


def test_lowercases_and_strips_punctuation():
    assert tokenize("EBITDA?") == ["ebitda"]
    assert tokenize("West Container Terminal status?") == ["west", "container", "terminal", "statu"]


def test_numbers_keep_decimals_and_percent():
    assert tokenize("Rs.80.01 billion, a rise of 75%") == ["rs", "80.01", "billion", "rise", "75%"]


def test_stopwords_removed():
    assert tokenize("What is the profit of the Group?") == ["profit", "group"]


def test_stemming_matches_word_forms():
    # The real failure this fixed: "manage" in the question, "management" in the report.
    assert tokenize("manage") == tokenize("management")
    assert tokenize("rooms") == tokenize("room")
    assert tokenize("increased") == tokenize("increases")
    assert tokenize("companies") == ["company"]


def test_short_words_and_numbers_not_stemmed():
    assert tokenize("bus 2025 gas") == ["bus", "2025", "gas"]
