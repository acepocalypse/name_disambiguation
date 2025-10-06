from name_matcher.canonical import generate_clean_canonical


def test_generate_clean_canonical_prefers_full_name():
    names = [
        "J. B. Smith",
        "John Barrett Smith",
        "John Smith",
    ]
    assert generate_clean_canonical(names) == "John Barrett Smith"


def test_generate_clean_canonical_handles_empty_cluster():
    assert generate_clean_canonical([]) == ""
