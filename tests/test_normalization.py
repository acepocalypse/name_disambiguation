import pytest

from name_matcher.normalization import normalize


def test_normalize_removes_titles_and_suffixes():
    assert normalize("Dr. John A. Smith, Jr.") == "john a smith"


def test_normalize_handles_last_name_first():
    assert normalize("Smith, John") == "john smith"


def test_normalize_handles_unicode():
    assert normalize("Jos\u00e9 \u00c1lvarez") == "jose alvarez"


def test_normalize_empty_string():
    assert normalize("") == ""
